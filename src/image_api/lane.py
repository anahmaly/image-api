from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class LaneBusy(RuntimeError):
    pass


class GpuLane:
    """Cross-process singleton lane backed by an advisory filesystem lock."""

    def __init__(self, path: Path, timeout_seconds: float = 2.0) -> None:
        self.path = path
        self.status_path = path.with_suffix(path.suffix + ".status")
        self.timeout_seconds = timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquire(self, capability: str) -> Iterator[None]:
        """Acquire the lane within the configured bounded admission timeout."""
        with self._acquire(capability, timeout_seconds=self.timeout_seconds):
            yield

    @contextmanager
    def acquire_wait(self, capability: str) -> Iterator[None]:
        """Wait until the lane is available for already-admitted durable work."""
        with self._acquire(capability, timeout_seconds=None):
            yield

    @contextmanager
    def _acquire(self, capability: str, *, timeout_seconds: float | None) -> Iterator[None]:
        handle = self.path.open("a+b")
        acquired = False
        try:
            if timeout_seconds is None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                acquired = True
            else:
                deadline = time.monotonic() + timeout_seconds
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except BlockingIOError as exc:
                        if time.monotonic() >= deadline:
                            raise LaneBusy("GPU lane is busy") from exc
                        time.sleep(0.01)
            temporary = self.status_path.with_suffix(
                self.status_path.suffix + f".{os.getpid()}.tmp"
            )
            temporary.write_text(json.dumps({"activeCapability": capability, "active": True}))
            os.replace(temporary, self.status_path)
            yield
        finally:
            try:
                if acquired:
                    self.status_path.write_text(
                        json.dumps({"activeCapability": None, "active": False})
                    )
            finally:
                try:
                    if acquired:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()

    def status(self) -> dict[str, object]:
        handle = self.path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            return {"activeCapability": None, "active": False}
        try:
            value = json.loads(self.status_path.read_text())
            if (
                isinstance(value, dict)
                and isinstance(value.get("active"), bool)
                and (
                    value.get("activeCapability") is None
                    or isinstance(value.get("activeCapability"), str)
                )
            ):
                return {
                    "activeCapability": value.get("activeCapability"),
                    "active": value["active"],
                }
        except (OSError, ValueError, TypeError):
            pass
        return {"activeCapability": None, "active": False}
