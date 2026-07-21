from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from image_api.generation import (
    GenerationRunner,
    recover_interrupted_tasks,
    start_worker_heartbeat,
)
from image_api.lane import GpuLane
from image_api.store import TaskStore
from image_api_workers.ideogram import IdeogramModel

logging.basicConfig(level=os.getenv("IMAGE_API_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    state = Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
    start_worker_heartbeat(state / "generation-worker.heartbeat")
    store = TaskStore(state / "tasks.sqlite3", int(os.getenv("IMAGE_API_MAX_QUEUE_DEPTH", "100")))
    recovered = recover_interrupted_tasks(store, state / "outputs")
    if recovered:
        logger.warning("Reconciled interrupted generation tasks: count=%s", recovered)
    runner = GenerationRunner(
        store,
        GpuLane(state / "gpu-lane.lock", float(os.getenv("IMAGE_API_LANE_TIMEOUT_SECONDS", "120"))),
        state / "outputs",
        IdeogramModel(
            Path(os.getenv("IMAGE_API_IDEOGRAM_WEIGHTS_PATH", "/models/ideogram-4-nf4")),
            status_path=state / "generation-model-status.json",
        ),
    )
    poll = float(os.getenv("IMAGE_API_GENERATION_POLL_SECONDS", "0.5"))
    while True:
        if not runner.run_one():
            time.sleep(poll)


if __name__ == "__main__":
    main()
