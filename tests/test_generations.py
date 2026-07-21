from __future__ import annotations

import concurrent.futures

import pytest
from fastapi.testclient import TestClient

from image_api.app import create_app
from image_api.config import Settings
from image_api.store import IdempotencyConflict, TaskStore
from image_api.workers import FakeWorkerClient


def valid_request(**changes):
    value = {
        "width": 512,
        "height": 768,
        "seed": 42,
        "sampler_preset": "V4_DEFAULT_20",
        "structured_caption": {"description": "a blue ceramic bee", "style": {"type": "photo"}},
    }
    value.update(changes)
    return value


@pytest.fixture
def setup(tmp_path):
    settings = Settings.for_tests(tmp_path)
    store = TaskStore(settings.database_path)
    worker = FakeWorkerClient()
    client = TestClient(create_app(settings=settings, store=store, workers=worker))
    return client, store, worker


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 255},
        {"width": 513},
        {"height": 2064},
        {"seed": -1},
        {"sampler_preset": "made-up"},
        {"structured_caption": {}},
        {"structured_caption": None, "prompt": "plain", "magic_prompt": False},
    ],
)
def test_generation_schema_rejects_before_admission(setup, changes) -> None:
    client, store, _ = setup
    response = client.post(
        "/v1/generations", json=valid_request(**changes), headers={"Idempotency-Key": "schema-1"}
    )
    assert response.status_code == 422
    assert store.count() == 0


def test_plain_prompt_requires_configured_magic_prompt_backend(tmp_path) -> None:
    settings = Settings.for_tests(tmp_path, magic_prompt_backend=None)
    store = TaskStore(settings.database_path)
    client = TestClient(create_app(settings=settings, store=store, workers=FakeWorkerClient()))
    response = client.post(
        "/v1/generations",
        json=valid_request(structured_caption=None, prompt="a bee", magic_prompt=True),
        headers={"Idempotency-Key": "magic-1"},
    )
    assert response.status_code == 422
    assert store.count() == 0


def test_admission_is_durable_before_202_and_replay_returns_same_task(setup) -> None:
    client, store, _ = setup
    first = client.post(
        "/v1/generations", json=valid_request(), headers={"Idempotency-Key": "stable-key"}
    )
    second = client.post(
        "/v1/generations", json=valid_request(), headers={"Idempotency-Key": "stable-key"}
    )
    assert first.status_code == second.status_code == 202
    assert first.json() == second.json()
    task_id = first.json()["taskId"]
    assert store.get(task_id).status == "queued"
    assert "structured_caption" not in first.text


def test_same_key_different_fingerprint_conflicts(setup) -> None:
    client, store, _ = setup
    assert (
        client.post(
            "/v1/generations", json=valid_request(), headers={"Idempotency-Key": "conflict"}
        ).status_code
        == 202
    )
    response = client.post(
        "/v1/generations", json=valid_request(seed=43), headers={"Idempotency-Key": "conflict"}
    )
    assert response.status_code == 409
    assert store.count() == 1


def test_concurrent_same_key_admits_one_task(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    request = valid_request()

    def admit():
        return store.admit("same-key", request)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        tasks = list(pool.map(lambda _: admit(), range(24)))
    assert len({task.task_id for task in tasks}) == 1
    assert store.count() == 1


def test_store_detects_direct_conflicting_fingerprint(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.admit("key", valid_request())
    with pytest.raises(IdempotencyConflict):
        store.admit("key", valid_request(seed=99))
