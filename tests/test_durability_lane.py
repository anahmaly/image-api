from __future__ import annotations

import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from helpers import png
from image_api.generation import (
    GenerationRunner,
    worker_heartbeat_alive,
    write_worker_heartbeat,
)
from image_api.lane import GpuLane, LaneBusy
from image_api.store import TaskStore


def _hold_lane(path: str, entered, release) -> None:
    with GpuLane(Path(path), timeout_seconds=1).acquire("generation"):
        entered.set()
        release.wait(2)


def request():
    return {
        "width": 256,
        "height": 256,
        "seed": 1,
        "sampler_preset": "V4_TURBO_12",
        "structured_caption": {"description": "bee"},
    }


def test_cross_process_file_lane_is_singleton(tmp_path) -> None:
    first = GpuLane(tmp_path / "gpu.lock", timeout_seconds=0.05)
    second = GpuLane(tmp_path / "gpu.lock", timeout_seconds=0.05)
    with first.acquire("upscale"):
        try:
            with second.acquire("background-removal"):
                raise AssertionError("second lane acquisition must not succeed")
        except LaneBusy:
            pass
    with second.acquire("generation"):
        assert second.status()["activeCapability"] == "generation"


def test_gpu_lane_excludes_a_separate_process(tmp_path) -> None:
    path = tmp_path / "gpu.lock"
    entered = multiprocessing.Event()
    release = multiprocessing.Event()
    child = multiprocessing.Process(target=_hold_lane, args=(str(path), entered, release))
    child.start()
    try:
        assert entered.wait(2)
        with pytest.raises(LaneBusy):
            with GpuLane(path, timeout_seconds=0.05).acquire("upscale"):
                pass
    finally:
        release.set()
        child.join(2)
        if child.is_alive():
            child.terminate()
    assert child.exitcode == 0


def test_claim_is_exactly_once_under_concurrency(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.admit("one", request())

    def claim():
        claimed = store.claim_next("worker-a")
        return claimed.task_id if claimed else None

    threads = []
    results = []
    lock = threading.Lock()
    for _ in range(8):
        thread = threading.Thread(
            target=lambda: (lock.acquire(), results.append(claim()), lock.release())
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(task.task_id) == 1
    assert sum(item is not None for item in results) == 1


def test_restart_recovers_queued_and_conservatively_fails_running(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    queued = store.admit("queued", request())
    running = store.admit("running", request() | {"seed": 2})
    assert store.claim_next("dead-worker").task_id == queued.task_id
    changed = store.recover_after_restart()
    assert changed == 1
    assert store.get(queued.task_id).status == "failed"
    assert store.get(queued.task_id).error_code == "worker_interrupted"
    assert store.get(running.task_id).status == "queued"


def test_runner_invokes_once_and_atomically_publishes_valid_png(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.admit("run", request())
    calls = 0

    def model(body):
        nonlocal calls
        calls += 1
        return png(size=(body["width"], body["height"]))

    runner = GenerationRunner(store, GpuLane(tmp_path / "gpu.lock"), tmp_path / "outputs", model)
    assert runner.run_one() is True
    assert runner.run_one() is False
    complete = store.get(task.task_id)
    assert complete.status == "succeeded"
    assert calls == 1
    assert complete.image_name == f"{task.task_id}.png"
    assert not list((tmp_path / "outputs").glob("*.tmp"))
    with Image.open(tmp_path / "outputs" / complete.image_name) as image:
        assert image.size == (256, 256)
        image.verify()


def test_crash_after_claim_is_never_reinvoked(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.admit("crash", request())
    assert store.claim_next("crashed").task_id == task.task_id
    store.recover_after_restart()
    calls = 0

    def model(_):
        nonlocal calls
        calls += 1
        return png(size=(256, 256))

    runner = GenerationRunner(store, GpuLane(tmp_path / "gpu.lock"), tmp_path / "out", model)
    assert runner.run_one() is False
    assert calls == 0


def test_generation_worker_heartbeat_is_bounded_and_stale_safe(tmp_path) -> None:
    heartbeat = tmp_path / "state" / "generation-worker.heartbeat"
    write_worker_heartbeat(heartbeat)
    assert worker_heartbeat_alive(heartbeat, max_age_seconds=15) is True
    stale = time.time() - 30
    os.utime(heartbeat, (stale, stale))
    assert worker_heartbeat_alive(heartbeat, max_age_seconds=15) is False
