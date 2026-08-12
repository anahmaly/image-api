from __future__ import annotations

from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient

from image_api.app import create_app
from image_api.config import Settings
from image_api.workers import FakeWorkerClient, HttpWorkerClient


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


def test_flux_2_klein_generation_returns_png_for_the_exact_plain_prompt(client: TestClient) -> None:
    response = client.post(
        "/v1/generations",
        json={
            "model": "flux-2-klein-4b",
            "width": 256,
            "height": 256,
            "seed": 42,
            "prompt": "exact caller prompt",
            "magic_prompt": False,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_gateway_health_uses_configured_worker_timeout_for_generation_projection(tmp_path) -> None:
    generation_health = {
        "ready": True,
        "loaded": False,
        "device": "cuda",
        "weightsAvailable": True,
        "models": {
            "ideogram-4-nf4": {"weightsAvailable": True},
            "longcat-image-edit": {"weightsAvailable": True},
            "longcat-image-edit-turbo": {"weightsAvailable": True},
            "flux-2-klein-4b": {"weightsAvailable": True},
        },
    }

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.host == "generation" and request.extensions["timeout"]["read"] < 3:
            raise httpx.ReadTimeout("fixture rejects the former 0.25-second health timeout")
        return httpx.Response(
            200,
            json=generation_health
            if request.url.host == "generation"
            else {"ready": True, "loaded": False},
            request=request,
        )

    def gateway_health(timeout_seconds: float) -> dict[str, object]:
        workers = HttpWorkerClient(
            "http://upscale",
            "http://background",
            timeout_seconds,
            1_000_000,
            httpx.MockTransport(transport),
            "http://generation",
        )
        payload = (
            TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=workers))
            .get("/health")
            .json()
        )
        assert isinstance(payload, dict)
        return cast(dict[str, object], payload)

    def capability(health: dict[str, object], name: str) -> dict[str, object]:
        capabilities = health["capabilities"]
        assert isinstance(capabilities, dict)
        status = capabilities[name]
        assert isinstance(status, dict)
        return cast(dict[str, object], status)

    former = gateway_health(0.25)
    corrected = gateway_health(3.0)

    assert capability(former, "generation") == {
        "ready": False,
        "loaded": False,
        "device": "unavailable",
        "weightsAvailable": False,
        "workerAvailable": False,
        "models": {
            "ideogram-4-nf4": {"weightsAvailable": False, "ready": False},
            "longcat-image-edit": {"weightsAvailable": False, "ready": False},
            "longcat-image-edit-turbo": {"weightsAvailable": False, "ready": False},
            "flux-2-klein-4b": {"weightsAvailable": False, "ready": False},
        },
    }
    assert capability(corrected, "generation") == {
        "ready": True,
        "loaded": False,
        "device": "cuda",
        "weightsAvailable": True,
        "workerAvailable": True,
        "models": {
            "ideogram-4-nf4": {"weightsAvailable": True, "ready": True},
            "longcat-image-edit": {"weightsAvailable": True, "ready": True},
            "longcat-image-edit-turbo": {"weightsAvailable": True, "ready": True},
            "flux-2-klein-4b": {"weightsAvailable": True, "ready": True},
        },
    }
    assert capability(corrected, "upscale") == {
        "ready": True,
        "loaded": False,
        "device": "unavailable",
    }
    assert capability(corrected, "background-removal") == {
        "ready": True,
        "loaded": False,
        "device": "unavailable",
    }


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
