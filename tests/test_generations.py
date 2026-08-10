from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from image_api.app import create_app
from image_api.config import Settings
from image_api.workers import FakeWorkerClient


def valid_request(**changes: object) -> dict[str, object]:
    request: dict[str, object] = {
        "width": 256,
        "height": 256,
        "seed": 42,
        "sampler_preset": "V4_DEFAULT_20",
        "structured_caption": {"description": "a blue ceramic bee"},
    }
    request.update(changes)
    return request


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=FakeWorkerClient()))


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 255},
        {"height": 2064},
        {"seed": -1},
        {"sampler_preset": "made-up"},
        {"structured_caption": {}},
        {"structured_caption": None, "prompt": "plain", "magic_prompt": False},
    ],
)
def test_generation_schema_rejects_before_direct_worker_dispatch(
    client: TestClient, changes
) -> None:
    assert client.post("/v1/generations", json=valid_request(**changes)).status_code == 422


def test_generation_returns_png_synchronously(client: TestClient) -> None:
    response = client.post("/v1/generations", json=valid_request())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_plain_prompt_requires_configured_magic_prompt_backend(tmp_path) -> None:
    client = TestClient(
        create_app(
            settings=Settings.for_tests(tmp_path, magic_prompt_backend=None),
            workers=FakeWorkerClient(),
        )
    )
    response = client.post(
        "/v1/generations",
        json=valid_request(structured_caption=None, prompt="a bee", magic_prompt=True),
    )
    assert response.status_code == 422
