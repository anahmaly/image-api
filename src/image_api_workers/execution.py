from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, TypeVar

from image_api.lane import GpuLane

T = TypeVar("T")


def execute_in_gpu_lane(
    capability: str,
    operation: Callable[[], T],
    *,
    state_dir: Path | None = None,
    timeout_seconds: float | None = None,
) -> T:
    """Hold the shared worker execution lane until GPU work and postprocessing finish."""
    state = state_dir or Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.getenv("IMAGE_API_LANE_TIMEOUT_SECONDS", "120"))
    )
    with GpuLane(state / "gpu-lane.lock", timeout).acquire(capability):
        return operation()
