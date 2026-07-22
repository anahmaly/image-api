from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import stat
from pathlib import Path
from typing import Iterator

import fcntl

logger = logging.getLogger(__name__)
STATE_UID = 10001
STATE_GID = 10001
DEFAULT_MAX_ENTRIES = 100_000
SOURCE_LOCK_NAME = ".source-files.lock"
STATE_INIT_LOCK_NAME = ".state-init.lock"
READINESS_FILE_NAME = ".write-readiness"


class StateMigrationLimit(RuntimeError):
    pass


def _state_entries(root: Path, max_entries: int) -> Iterator[Path]:
    pending = [root]
    seen = 0
    while pending:
        path = pending.pop()
        seen += 1
        if seen > max_entries:
            raise StateMigrationLimit("state migration entry limit exceeded")
        yield path
        if path.is_symlink() or not path.is_dir():
            continue
        with os.scandir(path) as entries:
            pending.extend(Path(entry.path) for entry in entries)


def initialize_state(
    state_dir: Path,
    *,
    uid: int = STATE_UID,
    gid: int = STATE_GID,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """Bound ownership migration without following links or crossing the state volume."""
    if max_entries < 1:
        raise ValueError("state migration entry limit must be positive")
    if state_dir.is_symlink():
        raise ValueError("state directory must not be a symbolic link")
    state_dir.mkdir(parents=True, exist_ok=True)
    init_lock_path = state_dir / STATE_INIT_LOCK_NAME
    with init_lock_path.open("a+b") as init_lock:
        try:
            fcntl.flock(init_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("state migration is already running") from exc
        root_device = state_dir.stat().st_dev
        for path in _state_entries(state_dir, max_entries):
            metadata = path.lstat()
            if metadata.st_dev != root_device or stat.S_ISLNK(metadata.st_mode):
                continue
            os.chown(path, uid, gid, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                path.chmod(metadata.st_mode | stat.S_IRWXU, follow_symlinks=False)
            elif stat.S_ISREG(metadata.st_mode):
                path.chmod(metadata.st_mode | stat.S_IRUSR | stat.S_IWUSR, follow_symlinks=False)
        fcntl.flock(init_lock.fileno(), fcntl.LOCK_UN)


def state_write_ready(state_dir: Path, database_path: Path, source_dir: Path) -> bool:
    """Exercise shared source locking/file publication and a committed SQLite write."""
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        lock_path = source_dir / SOURCE_LOCK_NAME
        marker_path = source_dir / READINESS_FILE_NAME
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with marker_path.open("w+b") as marker:
                    marker.write(b"ready")
                    marker.flush()
                    os.fsync(marker.fileno())
                marker_path.unlink()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        with sqlite3.connect(database_path, timeout=5) as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS state_readiness (singleton INTEGER PRIMARY KEY)"
            )
            connection.execute("INSERT OR REPLACE INTO state_readiness(singleton) VALUES (1)")
            connection.execute("DELETE FROM state_readiness WHERE singleton=1")
            connection.commit()
        return True
    except (OSError, sqlite3.Error) as exc:
        logger.error(
            "shared state write readiness failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "check"))
    args = parser.parse_args()
    state = Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
    if args.command == "init":
        initialize_state(
            state,
            uid=int(os.getenv("IMAGE_API_STATE_UID", str(STATE_UID))),
            gid=int(os.getenv("IMAGE_API_STATE_GID", str(STATE_GID))),
            max_entries=int(
                os.getenv("IMAGE_API_STATE_INIT_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES))
            ),
        )
        return
    if not state_write_ready(state, state / "tasks.sqlite3", state / "sources"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
