from __future__ import annotations

import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from image_api.store import (
    IdempotencyConflict,
    PersistedOutputQuotaExceeded,
    TaskStore,
)


def _initialize_with_snapshot_barrier(
    database_path: str,
    snapshot_barrier: Any,
    results: Any,
) -> None:
    class SnapshotBarrierConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            if sql.startswith("PRAGMA table_info") and not self.in_transaction:
                snapshot_barrier.wait(timeout=5)
            return super().execute(sql, parameters)

    class BarrierTaskStore(TaskStore):
        def _connect(self) -> sqlite3.Connection:
            connection = sqlite3.connect(
                self.path,
                timeout=10,
                isolation_level=None,
                factory=SnapshotBarrierConnection,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            return connection

    try:
        BarrierTaskStore(Path(database_path))
    except Exception as exc:
        results.put((False, type(exc).__name__, str(exc)))
    else:
        results.put((True, None, None))


def test_concurrent_processes_serialize_legacy_schema_migration(tmp_path: Path) -> None:
    database_path = tmp_path / "tasks.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE generation_tasks (
                task_id TEXT PRIMARY KEY,
                idempotency_hash TEXT NOT NULL UNIQUE,
                fingerprint TEXT NOT NULL,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
                worker_id TEXT,
                error_code TEXT,
                image_name TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """INSERT INTO generation_tasks
               (task_id,idempotency_hash,fingerprint,request_json,status,created_at,updated_at)
               VALUES ('legacy-task','key-hash','fingerprint','{}','succeeded',1,2)"""
        )

    context = multiprocessing.get_context("fork")
    snapshot_barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_initialize_with_snapshot_barrier,
            args=(str(database_path), snapshot_barrier, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(2)

    observed = [results.get(timeout=2) for _ in processes]
    assert observed == [(True, None, None), (True, None, None)]
    assert [process.exitcode for process in processes] == [0, 0]

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]: (row[2], row[3], row[4])
            for row in connection.execute("PRAGMA table_info(generation_tasks)")
        }
        legacy = connection.execute(
            """SELECT task_id,status,created_at,updated_at,task_kind,
                      output_sha256,output_width,output_height,output_mode,
                      output_reserved_bytes,output_size_bytes
               FROM generation_tasks WHERE task_id='legacy-task'"""
        ).fetchone()
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(generation_tasks)")}

    assert columns == {
        "task_id": ("TEXT", 0, None),
        "idempotency_hash": ("TEXT", 1, None),
        "fingerprint": ("TEXT", 1, None),
        "request_json": ("TEXT", 1, None),
        "status": ("TEXT", 1, None),
        "worker_id": ("TEXT", 0, None),
        "error_code": ("TEXT", 0, None),
        "image_name": ("TEXT", 0, None),
        "created_at": ("INTEGER", 1, None),
        "updated_at": ("INTEGER", 1, None),
        "task_kind": ("TEXT", 1, "'generation'"),
        "output_sha256": ("TEXT", 0, None),
        "output_width": ("INTEGER", 0, None),
        "output_height": ("INTEGER", 0, None),
        "output_mode": ("TEXT", 0, None),
        "output_reserved_bytes": ("INTEGER", 0, None),
        "output_size_bytes": ("INTEGER", 0, None),
    }
    assert legacy == (
        "legacy-task",
        "succeeded",
        1,
        2,
        "generation",
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert "generation_tasks_capability_claim" in indexes


def processing_request(seed: int = 1) -> dict[str, object]:
    return {
        "task_type": "upscale",
        "seed": seed,
        "source_image_name": f"{'a' * 64}-{'b' * 64}.png",
    }


def quota_store(tmp_path: Path, *, quota: int = 10, ceiling: int = 6) -> TaskStore:
    return TaskStore(
        tmp_path / "tasks.sqlite3",
        processing_max_persisted_output_bytes=quota,
        processing_max_encoded_output_bytes=ceiling,
        output_dir=tmp_path / "outputs",
    )


def test_processing_replay_and_conflict_precede_full_quota_check(tmp_path: Path) -> None:
    store = quota_store(tmp_path, quota=6, ceiling=6)
    admitted = store.admit("same-key", processing_request(), "upscale")

    assert store.admit("same-key", processing_request(), "upscale") == admitted
    with pytest.raises(IdempotencyConflict):
        store.admit("same-key", processing_request(seed=2), "upscale")
    with pytest.raises(PersistedOutputQuotaExceeded):
        store.admit("other-key", processing_request(seed=3), "upscale")
    assert store.count() == 1


def test_concurrent_processing_admissions_cannot_overbook_quota(tmp_path: Path) -> None:
    first = quota_store(tmp_path)
    second = quota_store(tmp_path)

    def admit(store: TaskStore, key: str, seed: int) -> str:
        try:
            return store.admit(key, processing_request(seed), "upscale").task_id
        except PersistedOutputQuotaExceeded:
            return "quota-full"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda arguments: admit(*arguments),
                [(first, "first", 1), (second, "second", 2)],
            )
        )

    assert outcomes.count("quota-full") == 1
    assert sum(outcome != "quota-full" for outcome in outcomes) == 1
    assert first.processing_output_bytes_used() == 6
    assert first.count() == 1


def test_processing_quota_uses_reservation_then_exact_success_size(tmp_path: Path) -> None:
    store = quota_store(tmp_path)
    first = store.admit("first", processing_request(), "upscale")
    assert store.processing_output_bytes_used() == 6
    claimed = store.claim_next("worker", "upscale")
    assert claimed is not None and claimed.task_id == first.task_id
    assert store.processing_output_bytes_used() == 6

    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / f"{first.task_id}.png").write_bytes(b"four")
    store.succeed(first.task_id, f"{first.task_id}.png", output_size_bytes=4)
    assert store.processing_output_bytes_used() == 4

    second = store.admit("second", processing_request(seed=2), "background-removal")
    assert second.task_kind == "background-removal"
    assert store.processing_output_bytes_used() == 10


def test_processing_publication_cannot_exceed_its_reservation(tmp_path: Path) -> None:
    store = quota_store(tmp_path, quota=20, ceiling=6)
    task = store.admit("oversized", processing_request(), "upscale")
    claimed = store.claim_next("worker", "upscale")
    assert claimed is not None and claimed.task_id == task.task_id

    with pytest.raises(ValueError, match="reservation"):
        store.succeed(task.task_id, f"{task.task_id}.png", output_size_bytes=7)

    assert store.get(task.task_id).status == "running"
    assert store.processing_output_bytes_used() == 6


def test_failed_processing_task_releases_reservation(tmp_path: Path) -> None:
    store = quota_store(tmp_path, quota=6, ceiling=6)
    first = store.admit("first", processing_request(), "upscale")
    claimed = store.claim_next("worker", "upscale")
    assert claimed is not None and claimed.task_id == first.task_id
    store.fail(first.task_id, "bounded_failure")

    replacement = store.admit("replacement", processing_request(seed=2), "upscale")
    assert replacement.task_id != first.task_id
    assert store.processing_output_bytes_used() == 6


def test_generation_tasks_are_excluded_from_processing_quota(tmp_path: Path) -> None:
    store = quota_store(tmp_path, quota=6, ceiling=6)
    store.admit("processing", processing_request(), "upscale")

    generation = store.admit("generation", {"width": 1, "height": 1})

    assert generation.task_kind == "generation"
    assert store.processing_output_bytes_used() == 6


def test_legacy_succeeded_processing_rows_use_file_size_or_reserved_ceiling(
    tmp_path: Path,
) -> None:
    store = quota_store(tmp_path, quota=100, ceiling=6)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    with sqlite3.connect(store.path) as connection:
        for task_id, seed, output_size in (
            ("with-file", 1, None),
            ("missing-file", 2, None),
            ("tampered-size", 3, 1),
        ):
            connection.execute(
                """INSERT INTO generation_tasks
                   (task_id,idempotency_hash,fingerprint,request_json,status,image_name,task_kind,
                    created_at,updated_at,output_reserved_bytes,output_size_bytes)
                   VALUES (?,?,?,?, 'succeeded',?,'upscale',1,1,NULL,?)""",
                (
                    task_id,
                    f"key-{seed}",
                    f"fingerprint-{seed}",
                    TaskStore._canonical(processing_request(seed)),
                    f"{task_id}.png",
                    output_size,
                ),
            )
    (output_dir / "with-file.png").write_bytes(b"four")
    (output_dir / "tampered-size.png").write_bytes(b"four")

    restarted = quota_store(tmp_path, quota=100, ceiling=6)

    assert restarted.processing_output_bytes_used() == 14
