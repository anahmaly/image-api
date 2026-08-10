from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


def execute_in_gpu_lane(
    capability: str,
    operation: Callable[[], T],
) -> T:
    """Execute after the gateway has admitted the one global operation."""
    del capability
    return operation()
