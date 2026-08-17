from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar


class CoordinatorBusy(RuntimeError):
    """The single gateway-owned execution slot is occupied."""


T = TypeVar("T")


class SingleFlightCoordinator:
    """One in-memory execution authority for the one production gateway."""

    def __init__(self) -> None:
        self._slot = threading.Lock()
        self._active = 0
        self._owner: int | None = None
        self._guard = threading.Lock()

    def run(self, operation: Callable[[], T]) -> T:
        owner = threading.get_ident()
        with self._guard:
            if self._owner == owner:
                raise CoordinatorBusy("image execution is already active")
        self._slot.acquire()
        with self._guard:
            self._active = 1
            self._owner = owner
        try:
            return operation()
        finally:
            with self._guard:
                self._active = 0
                self._owner = None
            self._slot.release()

    def status(self) -> dict[str, object]:
        with self._guard:
            return {"ready": True, "active": self._active, "capacity": 1}
