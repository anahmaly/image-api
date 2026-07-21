from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient


@pytest.fixture
def setup(tmp_path):
    settings = Settings.for_tests(tmp_path)
    worker = FakeWorkerClient()
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=worker)
    )
    return client, worker


@pytest.mark.parametrize("model", ["bria-rmbg-2.0", "birefnet-hr-matting"])
def test_background_dispatch_returns_same_size_rgba(setup, model: str) -> None:
    client, worker = setup
    response = client.post(
        f"/v1/background-removal?model={model}&alpha_blur=1.5&alpha_threshold=4",
        files={"file": ("input.png", png("RGB", (11, 5)), "image/png")},
    )
    assert response.status_code == 200
    with Image.open(BytesIO(response.content)) as image:
        assert image.mode == "RGBA"
        assert image.size == (11, 5)
    assert worker.last_background["model"] == model
    assert worker.last_background["alpha_blur"] == 1.5


def test_background_options_are_bounded_before_dispatch(setup) -> None:
    client, worker = setup
    response = client.post(
        "/v1/background-removal?model=birefnet-hr-matting&birefnet_inference_size=4097",
        files={"file": ("input.png", png(), "image/png")},
    )
    assert response.status_code == 422
    assert worker.model_invocations == 0


def test_unknown_background_model_is_rejected(setup) -> None:
    client, worker = setup
    response = client.post(
        "/v1/background-removal?model=not-real",
        files={"file": ("input.png", png(), "image/png")},
    )
    assert response.status_code == 422
    assert worker.model_invocations == 0


def test_invalid_backend_image_is_distinct_safe_gateway_failure(tmp_path) -> None:
    class InvalidOutputWorker(FakeWorkerClient):
        def background(self, data: bytes, **parameters: object) -> bytes:
            self.model_invocations += 1
            return b"not-a-png"

    settings = Settings.for_tests(tmp_path)
    client = TestClient(
        create_app(
            settings=settings,
            store=TaskStore(settings.database_path),
            workers=InvalidOutputWorker(),
        )
    )
    response = client.post(
        "/v1/background-removal?model=birefnet-hr-matting",
        files={"file": ("input.png", png(), "image/png")},
    )
    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "invalid_worker_output",
            "message": "Image capability returned invalid output",
        }
    }
