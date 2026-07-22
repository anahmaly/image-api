from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pytest

from image_api.state import StateMigrationLimit, initialize_state, state_write_ready

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ("image-api", "upscale-worker", "background-worker", "generation-worker")


def test_legacy_root_owned_state_is_migrated_to_shared_numeric_identity(
    tmp_path: Path, monkeypatch
) -> None:
    state = tmp_path / "state"
    sources = state / "sources"
    sources.mkdir(parents=True)
    lock = sources / ".source-files.lock"
    lock.write_bytes(b"")
    database = state / "tasks.sqlite3"
    sqlite3.connect(database).close()
    outside = tmp_path / "outside"
    outside.touch()
    symlink = state / "outside-link"
    symlink.symlink_to(outside)
    ownership: list[tuple[Path, int, int]] = []
    monkeypatch.setattr(
        os,
        "chown",
        lambda path, uid, gid, *, dir_fd=None, follow_symlinks=True: ownership.append(
            (Path(path), uid, gid)
        ),
    )

    initialize_state(state, uid=10001, gid=10001, max_entries=100)

    assert (lock, 10001, 10001) in ownership
    assert (database, 10001, 10001) in ownership
    assert not any(path == symlink for path, _, _ in ownership)
    assert all((uid, gid) == (10001, 10001) for _, uid, gid in ownership)


def test_state_migration_is_bounded(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "one").touch()
    (state / "two").touch()
    monkeypatch.setattr(os, "chown", lambda *_args, **_kwargs: None)

    with pytest.raises(StateMigrationLimit):
        initialize_state(state, uid=10001, gid=10001, max_entries=2)


def test_state_readiness_proves_source_lock_file_and_sqlite_writes(tmp_path: Path) -> None:
    state = tmp_path / "state"

    assert state_write_ready(state, state / "tasks.sqlite3", state / "sources") is True
    assert (state / "sources" / ".source-files.lock").is_file()
    assert (state / "tasks.sqlite3").is_file()


def test_compose_uses_bounded_root_init_and_waits_before_every_shared_writer() -> None:
    compose = (ROOT / "compose.yml").read_text()

    assert "state-init:" in compose
    assert 'user: "0:0"' in compose
    assert "python -m image_api.state init" in compose
    assert "IMAGE_API_STATE_INIT_MAX_ENTRIES" in compose
    for service in SERVICES:
        match = re.search(rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  \S|\Z)", compose)
        assert match is not None
        service_block = match.group("body")
        assert 'user: "10001:10001"' in service_block
        assert "state-init:" in service_block
        assert "condition: service_completed_successfully" in service_block


def test_runtime_images_define_the_same_uid_and_gid() -> None:
    for name in (
        "Dockerfile.gateway",
        "Dockerfile.test",
        "Dockerfile.upscale",
        "Dockerfile.background",
        "Dockerfile.generation",
    ):
        text = (ROOT / name).read_text()
        assert "--gid 10001" in text
        assert "--uid 10001 --gid 10001" in text
        assert "USER 10001:10001" in text


def test_model_mounts_remain_read_only() -> None:
    compose = (ROOT / "compose.yml").read_text()
    model_mounts = [line for line in compose.splitlines() if ":/models/" in line]
    assert model_mounts
    assert all(line.rstrip().endswith(":ro") for line in model_mounts)
