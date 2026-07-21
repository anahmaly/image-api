from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from image_api.generation import GenerationRunner, start_worker_heartbeat
from image_api.lane import GpuLane
from image_api.store import TaskStore


def fake_model(request: dict[str, object]) -> bytes:
    width = request.get("width")
    height = request.get("height")
    if type(width) is not int or type(height) is not int:
        raise ValueError("invalid test generation dimensions")
    image = Image.new("RGB", (width, height), (20, 30, 40))
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def main() -> None:
    state = Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
    start_worker_heartbeat(state / "generation-worker.heartbeat")
    store = TaskStore(state / "tasks.sqlite3")
    store.recover_after_restart()
    runner = GenerationRunner(
        store, GpuLane(state / "gpu-lane.lock", 30), state / "outputs", fake_model
    )
    while True:
        if not runner.run_one():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
