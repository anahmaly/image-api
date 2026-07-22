from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal["queued", "running", "succeeded", "failed"]
TaskKind = Literal["generation", "upscale", "background-removal"]
TASK_KINDS: tuple[TaskKind, ...] = ("generation", "upscale", "background-removal")


class IdempotencyConflict(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


class PersistedOutputQuotaExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_kind: TaskKind
    status: TaskStatus
    request: dict[str, Any]
    error_code: str | None
    image_name: str | None
    output_sha256: str | None
    output_width: int | None
    output_height: int | None
    output_mode: str | None
    output_reserved_bytes: int | None
    output_size_bytes: int | None
    created_at: int
    updated_at: int


class TaskStore:
    def __init__(
        self,
        path: Path,
        max_queue_depth: int = 100,
        *,
        processing_max_persisted_output_bytes: int = 18_000_000_000,
        processing_max_encoded_output_bytes: int = 300_000_000,
        output_dir: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.max_queue_depth = max_queue_depth
        self.processing_max_persisted_output_bytes = processing_max_persisted_output_bytes
        self.processing_max_encoded_output_bytes = processing_max_encoded_output_bytes
        self.output_dir = (
            Path(output_dir) if output_dir is not None else self.path.parent / "outputs"
        )
        if (
            min(
                self.max_queue_depth,
                self.processing_max_persisted_output_bytes,
                self.processing_max_encoded_output_bytes,
            )
            < 1
        ):
            raise ValueError("task store limits must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            # Serialize the entire inspection/migration section across processes. SQLite DDL is
            # transactional, so another initializer observes either the old schema or all changes.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_tasks (
                    task_id TEXT PRIMARY KEY,
                    idempotency_hash TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
                    worker_id TEXT,
                    error_code TEXT,
                    image_name TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    task_kind TEXT NOT NULL DEFAULT 'generation',
                    output_sha256 TEXT,
                    output_width INTEGER,
                    output_height INTEGER,
                    output_mode TEXT,
                    output_reserved_bytes INTEGER,
                    output_size_bytes INTEGER
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(generation_tasks)").fetchall()
            }
            migrations = {
                "task_kind": "TEXT NOT NULL DEFAULT 'generation'",
                "output_sha256": "TEXT",
                "output_width": "INTEGER",
                "output_height": "INTEGER",
                "output_mode": "TEXT",
                "output_reserved_bytes": "INTEGER",
                "output_size_bytes": "INTEGER",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE generation_tasks ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS generation_tasks_capability_claim "
                "ON generation_tasks(task_kind,status,created_at)"
            )
            connection.commit()

    @staticmethod
    def _canonical(request: dict[str, Any]) -> str:
        return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _row(cls, row: sqlite3.Row) -> TaskRecord:
        task_kind = row["task_kind"]
        if task_kind not in TASK_KINDS:
            raise RuntimeError("persisted task has invalid kind")
        return TaskRecord(
            task_id=row["task_id"],
            task_kind=task_kind,
            status=row["status"],
            request=json.loads(row["request_json"]),
            error_code=row["error_code"],
            image_name=row["image_name"],
            output_sha256=row["output_sha256"],
            output_width=row["output_width"],
            output_height=row["output_height"],
            output_mode=row["output_mode"],
            output_reserved_bytes=row["output_reserved_bytes"],
            output_size_bytes=row["output_size_bytes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def admit(
        self,
        idempotency_key: str,
        request: dict[str, Any],
        task_kind: TaskKind = "generation",
    ) -> TaskRecord:
        canonical = self._canonical(request)
        fingerprint = self._hash(self._canonical({"task_kind": task_kind, "request": request}))
        key_hash = self._hash(idempotency_key)
        now = time.time_ns()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM generation_tasks WHERE idempotency_hash = ?", (key_hash,)
            ).fetchone()
            if existing is not None:
                connection.commit()
                if existing["fingerprint"] != fingerprint:
                    # Legacy rows used a request-only fingerprint. Preserve exact legacy replay.
                    legacy = self._hash(canonical)
                    if not (
                        existing["task_kind"] == "generation" and existing["fingerprint"] == legacy
                    ):
                        raise IdempotencyConflict(
                            "idempotency key already identifies another request"
                        )
                return self._row(existing)
            depth = connection.execute(
                "SELECT COUNT(*) FROM generation_tasks WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if depth >= self.max_queue_depth:
                connection.rollback()
                raise QueueFull("task queue is full")
            reservation: int | None = None
            if task_kind != "generation":
                reservation = self.processing_max_encoded_output_bytes
                used = self._processing_output_bytes_used(connection)
                if used + reservation > self.processing_max_persisted_output_bytes:
                    connection.rollback()
                    raise PersistedOutputQuotaExceeded("persisted processing output quota is full")
            task_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO generation_tasks
                   (task_id,idempotency_hash,fingerprint,request_json,status,task_kind,
                    created_at,updated_at,output_reserved_bytes)
                   VALUES (?,?,?,?, 'queued',?,?,?,?)""",
                (task_id, key_hash, fingerprint, canonical, task_kind, now, now, reservation),
            )
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            connection.commit()
            return self._row(row)

    def get(self, task_id: str) -> TaskRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._row(row)

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM generation_tasks").fetchone()[0])

    def _reservation_for_row(self, row: sqlite3.Row) -> int:
        reserved = row["output_reserved_bytes"]
        if type(reserved) is int and reserved > 0:
            return max(reserved, self.processing_max_encoded_output_bytes)
        return self.processing_max_encoded_output_bytes

    def _succeeded_output_bytes(self, row: sqlite3.Row) -> int:
        conservative = self._reservation_for_row(row)
        stored = row["output_size_bytes"]
        stored_size = stored if type(stored) is int and stored > 0 else 0
        image_name = row["image_name"]
        if image_name != f"{row['task_id']}.png":
            return max(stored_size, conservative)
        try:
            file_stat = (self.output_dir / image_name).lstat()
        except OSError:
            return max(stored_size, conservative)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size < 1:
            return max(stored_size, conservative)
        return max(stored_size, int(file_stat.st_size))

    def _processing_output_bytes_used(self, connection: sqlite3.Connection) -> int:
        rows = connection.execute(
            """SELECT task_id,status,image_name,output_reserved_bytes,output_size_bytes
               FROM generation_tasks
               WHERE task_kind IN ('upscale','background-removal')
                 AND status IN ('queued','running','succeeded')"""
        ).fetchall()
        return sum(
            self._succeeded_output_bytes(row)
            if row["status"] == "succeeded"
            else self._reservation_for_row(row)
            for row in rows
        )

    def processing_output_bytes_used(self) -> int:
        with self._connect() as connection:
            return self._processing_output_bytes_used(connection)

    def source_referenced(self, source_name: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM generation_tasks
                   WHERE status IN ('queued','running')
                     AND json_extract(request_json, '$.source_image_name') = ? LIMIT 1""",
                (source_name,),
            ).fetchone()
        return row is not None

    def active_source_names(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT json_extract(request_json, '$.source_image_name') AS source_name
                   FROM generation_tasks
                   WHERE status IN ('queued','running')
                     AND json_type(request_json, '$.source_image_name') = 'text'"""
            ).fetchall()
        return {str(row["source_name"]) for row in rows}

    def claim_next(self, worker_id: str, task_kind: TaskKind = "generation") -> TaskRecord | None:
        now = time.time_ns()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT task_id FROM generation_tasks
                   WHERE status='queued' AND task_kind=? ORDER BY created_at LIMIT 1""",
                (task_kind,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            task_id = row["task_id"]
            changed = connection.execute(
                """UPDATE generation_tasks SET status='running',worker_id=?,updated_at=?
                   WHERE task_id=? AND status='queued' AND task_kind=?""",
                (worker_id, now, task_id, task_kind),
            ).rowcount
            if changed != 1:
                connection.rollback()
                return None
            claimed = connection.execute(
                "SELECT * FROM generation_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            connection.commit()
            return self._row(claimed)

    def succeed(
        self,
        task_id: str,
        image_name: str,
        *,
        output_sha256: str | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        output_mode: str | None = None,
        output_size_bytes: int | None = None,
    ) -> None:
        self._transition(
            task_id,
            "succeeded",
            image_name=image_name,
            output_sha256=output_sha256,
            output_width=output_width,
            output_height=output_height,
            output_mode=output_mode,
            output_size_bytes=output_size_bytes,
        )

    def fail(self, task_id: str, error_code: str) -> None:
        self._transition(task_id, "failed", error_code=error_code[:64])

    def _transition(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error_code: str | None = None,
        image_name: str | None = None,
        output_sha256: str | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        output_mode: str | None = None,
        output_size_bytes: int | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT task_kind,status,output_reserved_bytes
                   FROM generation_tasks WHERE task_id=?""",
                (task_id,),
            ).fetchone()
            if row is None or row["status"] != "running":
                connection.rollback()
                raise RuntimeError("invalid task state transition")
            if status == "succeeded" and row["task_kind"] != "generation":
                reserved = row["output_reserved_bytes"]
                reservation = (
                    reserved
                    if type(reserved) is int and reserved > 0
                    else self.processing_max_encoded_output_bytes
                )
                publication_limit = min(reservation, self.processing_max_encoded_output_bytes)
                if (
                    type(output_size_bytes) is not int
                    or output_size_bytes < 1
                    or output_size_bytes > publication_limit
                ):
                    connection.rollback()
                    raise ValueError("processing output exceeds its durable reservation")
            changed = connection.execute(
                """UPDATE generation_tasks
                   SET status=?,error_code=?,image_name=?,output_sha256=?,output_width=?,
                       output_height=?,output_mode=?,output_size_bytes=?,worker_id=NULL,updated_at=?
                   WHERE task_id=? AND status='running'""",
                (
                    status,
                    error_code,
                    image_name,
                    output_sha256,
                    output_width,
                    output_height,
                    output_mode,
                    output_size_bytes,
                    time.time_ns(),
                    task_id,
                ),
            ).rowcount
            if changed != 1:
                connection.rollback()
                raise RuntimeError("invalid task state transition")
            connection.commit()

    def running(self, task_kind: TaskKind | None = None) -> list[TaskRecord]:
        with self._connect() as connection:
            if task_kind is None:
                rows = connection.execute(
                    "SELECT * FROM generation_tasks WHERE status='running' ORDER BY created_at"
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM generation_tasks
                       WHERE status='running' AND task_kind=? ORDER BY created_at""",
                    (task_kind,),
                ).fetchall()
        return [self._row(row) for row in rows]

    def reconcile_success(
        self,
        task_id: str,
        image_name: str,
        *,
        output_sha256: str | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        output_mode: str | None = None,
        output_size_bytes: int | None = None,
    ) -> bool:
        """Idempotently bind a validated canonical output to a running task."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT status,task_kind,image_name,output_sha256,output_width,output_height,
                          output_mode,output_reserved_bytes,output_size_bytes
                   FROM generation_tasks WHERE task_id=?""",
                (task_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            expected_metadata = (
                image_name,
                output_sha256,
                output_width,
                output_height,
                output_mode,
                output_size_bytes,
            )
            stored_metadata = (
                row["image_name"],
                row["output_sha256"],
                row["output_width"],
                row["output_height"],
                row["output_mode"],
                row["output_size_bytes"],
            )
            if row["status"] == "succeeded" and stored_metadata == expected_metadata:
                connection.commit()
                return True
            if row["status"] != "running":
                connection.rollback()
                return False
            if row["task_kind"] != "generation":
                reserved = row["output_reserved_bytes"]
                reservation = (
                    reserved
                    if type(reserved) is int and reserved > 0
                    else self.processing_max_encoded_output_bytes
                )
                publication_limit = min(reservation, self.processing_max_encoded_output_bytes)
                if (
                    type(output_size_bytes) is not int
                    or output_size_bytes < 1
                    or output_size_bytes > publication_limit
                ):
                    connection.rollback()
                    raise ValueError("processing output exceeds its durable reservation")
            connection.execute(
                """UPDATE generation_tasks
                   SET status='succeeded',error_code=NULL,image_name=?,output_sha256=?,
                       output_width=?,output_height=?,output_mode=?,output_size_bytes=?,
                       worker_id=NULL,updated_at=?
                   WHERE task_id=? AND status='running'""",
                (
                    image_name,
                    output_sha256,
                    output_width,
                    output_height,
                    output_mode,
                    output_size_bytes,
                    time.time_ns(),
                    task_id,
                ),
            )
            connection.commit()
            return True
