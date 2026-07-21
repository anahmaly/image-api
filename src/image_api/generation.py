from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from image_api.images import validate_png_output
from image_api.lane import GpuLane
from image_api.store import TaskStore

logger = logging.getLogger(__name__)
GenerationModel = Callable[[dict[str, object]], bytes]


def write_worker_heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def start_worker_heartbeat(path: Path, *, interval_seconds: float = 2.0) -> None:
    def heartbeat_loop() -> None:
        while True:
            try:
                write_worker_heartbeat(path)
            except OSError:
                safe_error = RuntimeError("state storage unavailable")
                logger.error(
                    "generation worker heartbeat failed",
                    exc_info=(type(safe_error), safe_error, safe_error.__traceback__),
                )
            time.sleep(interval_seconds)

    write_worker_heartbeat(path)
    threading.Thread(target=heartbeat_loop, name="generation-heartbeat", daemon=True).start()


def worker_heartbeat_alive(path: Path, *, max_age_seconds: float) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return 0 <= age <= max_age_seconds


class GenerationRunner:
    def __init__(
        self,
        store: TaskStore,
        lane: GpuLane,
        output_dir: Path,
        model: GenerationModel,
        worker_id: str | None = None,
    ) -> None:
        self.store = store
        self.lane = lane
        self.output_dir = output_dir
        self.model = model
        self.worker_id = worker_id or f"generation-{uuid.uuid4().hex[:12]}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_one(self) -> bool:
        task = self.store.claim_next(self.worker_id)
        if task is None:
            return False
        try:
            with self.lane.acquire("generation"):
                encoded = self.model(task.request)
            expected = (int(task.request["width"]), int(task.request["height"]))
            validate_png_output(
                encoded,
                expected_size=expected,
                required_mode=None,
                max_bytes=100_000_000,
                max_pixels=2048 * 2048,
            )
            image_name = f"{task.task_id}.png"
            final_path = self.output_dir / image_name
            temporary = self.output_dir / f".{task.task_id}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, final_path)
                directory_fd = os.open(self.output_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)
            self.store.succeed(task.task_id, image_name)
        except Exception:
            logger.exception("generation task failed")
            try:
                self.store.fail(task.task_id, "generation_failed")
            except Exception:
                logger.exception("generation task failure status could not be persisted")
        return True
