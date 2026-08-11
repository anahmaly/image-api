from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Callable
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.coordinator import SingleFlightCoordinator
from image_api.workers import FakeWorkerClient, HttpWorkerClient, WorkerUnavailable
from image_api_workers import background, upscale
from image_api_workers.generation_models import GenerationModels
from image_api_workers.generation_worker import create_worker_app


def test_global_single_flight_rejects_reentrant_request_and_releases_slot(tmp_path) -> None:
    coordinator = SingleFlightCoordinator()
    observed: list[int] = []

    def first() -> None:
        assert coordinator.status()["active"] == 1
        observed.append(1)
        try:
            coordinator.run(lambda: None)
        except Exception as exc:
            observed.append(int(exc.__class__.__name__ == "CoordinatorBusy"))

    coordinator.run(first)
    assert observed == [1, 1]
    assert coordinator.status() == {"ready": True, "active": 0, "capacity": 1}
    assert coordinator.run(lambda: "released") == "released"


def test_all_direct_capabilities_return_existing_synchronous_responses(tmp_path) -> None:
    worker = FakeWorkerClient()
    client = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=worker))
    source = {"file": ("input.png", png(), "image/png")}
    assert (
        client.post("/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=0", files=source)
        .headers["content-type"]
        .startswith("image/png")
    )
    assert (
        client.post("/v1/background-removal?model=birefnet-hr-matting", files=source)
        .headers["content-type"]
        .startswith("image/png")
    )
    assert (
        client.post(
            "/v1/generations",
            json={
                "width": 256,
                "height": 256,
                "seed": 1,
                "sampler_preset": "V4_TURBO_12",
                "structured_caption": {"description": "bee"},
            },
        )
        .headers["content-type"]
        .startswith("image/png")
    )
    assert worker.model_invocations == 3


def test_worker_unavailability_is_retryable_without_replay(tmp_path) -> None:
    class UnavailableWorker(FakeWorkerClient):
        def generation(self, request: dict[str, object]) -> bytes:
            self.model_invocations += 1
            raise WorkerUnavailable("not ready")

    worker = UnavailableWorker()
    client = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=worker))
    response = client.post(
        "/v1/generations",
        json={
            "width": 256,
            "height": 256,
            "seed": 1,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "bee"},
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "worker_unavailable"
    assert worker.model_invocations == 1


def test_public_routes_use_one_real_coordinator_and_internal_handlers_under_contention(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """One production-composed graph proves routing, contention, failure, and cleanup.

    The only fakes are model adapters beneath the actual internal worker handlers.
    ``entered``/``release`` are fixture-owned synchronous completion points: no
    elapsed-time polling, timeout, or production delay is involved.
    """

    manager = multiprocessing.Manager()

    class ModelBoundary:
        def __init__(self) -> None:
            self.calls = manager.list()
            self.state = manager.Namespace()
            self.state.failure = None
            self.state.unloads = 0
            self.hold = False
            self.entered = threading.Event()
            self.release = threading.Event()

        @property
        def failure(self) -> Exception | None:
            return self.state.failure

        @failure.setter
        def failure(self, value: Exception | None) -> None:
            self.state.failure = value

        @property
        def unloads(self) -> int:
            return self.state.unloads

        @unloads.setter
        def unloads(self, value: int) -> None:
            self.state.unloads = value

        def __call__(self, request: dict[str, object]) -> bytes:
            self.calls.append(request)
            if self.failure is not None:
                raise self.failure
            if self.hold:
                self.entered.set()
                self.release.wait()
            width = request.get("width", 8)
            height = request.get("height", 6)
            assert type(width) is int and type(height) is int
            return png(size=(width, height))

        def unload(self) -> None:
            self.unloads += 1

    class LongCatBoundary:
        loaded_model: str | None = None

        def __init__(self, model: ModelBoundary) -> None:
            self.model = model

        def __call__(self, request: dict[str, object]) -> bytes:
            model = request.get("model")
            assert model in {"longcat-image-edit", "longcat-image-edit-turbo"}
            self.loaded_model = str(model)
            source = request.get("source_image_bytes")
            assert isinstance(source, bytes)
            self.model.calls.append(request)
            if self.model.failure is not None:
                raise self.model.failure
            return source

        def unload(self) -> None:
            self.loaded_model = None
            self.model.unloads += 1

    boundary = ModelBoundary()
    models = GenerationModels(boundary, cast(Any, LongCatBoundary(boundary)))
    generation = TestClient(create_worker_app(models, Settings.for_tests(tmp_path)))
    upscale_app = TestClient(upscale.app)
    background_app = TestClient(background.app)
    dispatches: list[tuple[str, str, bytes]] = []
    active_models = 0
    max_active_models = 0

    def held_upscale(data: bytes, model: str, outscale: float, tile: int) -> bytes:
        nonlocal active_models, max_active_models
        assert model == "RealESRGAN_x4plus"
        assert outscale == 2.0
        assert tile == 0
        assert data == png()
        active_models += 1
        max_active_models = max(max_active_models, active_models)
        boundary.entered.set()
        boundary.release.wait()
        active_models -= 1
        return png(size=(16, 12))

    def fake_background(data: bytes, **_: object) -> bytes:
        assert data == png()
        return png("RGBA")

    monkeypatch.setattr(upscale, "_run", held_upscale)
    monkeypatch.setattr(background, "_run_background", fake_background)
    monkeypatch.setattr(background, "PeerEvictor", lambda _: lambda: None)

    clients = {
        "upscale": upscale_app,
        "background": background_app,
        "generation": generation,
    }

    def transport(request: httpx.Request) -> httpx.Response:
        capability = request.url.host
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "loaded": False,
                    "weightsAvailable": True,
                    "device": "cpu-test",
                    "models": {
                        "ideogram-4-nf4": {"weightsAvailable": True},
                        "longcat-image-edit": {"weightsAvailable": True},
                        "longcat-image-edit-turbo": {"weightsAvailable": True},
                    },
                },
                request=request,
            )
        dispatches.append((capability, request.url.raw_path.decode(), request.content))
        response = clients[capability].request(
            request.method,
            request.url.raw_path.decode(),
            content=request.content,
            headers={"content-type": request.headers.get("content-type", "")},
        )
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers=response.headers,
            request=request,
        )

    coordinator = SingleFlightCoordinator()
    workers = HttpWorkerClient(
        "http://upscale",
        "http://background",
        1,
        1_000_000,
        httpx.MockTransport(transport),
        "http://generation",
    )
    gateway = TestClient(
        create_app(settings=Settings.for_tests(tmp_path), workers=workers, coordinator=coordinator)
    )
    source = {"file": ("input.png", png(), "image/png")}

    held_result: list[object] = []

    def run_held_upscale() -> None:
        held_result.append(
            gateway.post("/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=0", files=source)
        )

    held = threading.Thread(target=run_held_upscale)
    held.start()
    boundary.entered.wait()
    assert coordinator.status()["active"] == 1
    assert dispatches == [
        (
            "upscale",
            "/internal/upscale?model=RealESRGAN_x4plus&outscale=2.0&tile=0",
            dispatches[0][2],
        )
    ]
    assert b'name="file"' in dispatches[0][2]
    assert png() in dispatches[0][2]

    busy_requests: tuple[Callable[[], object], ...] = (
        lambda: gateway.post(
            "/v1/generations",
            json={
                "width": 256,
                "height": 256,
                "seed": 7,
                "sampler_preset": "V4_TURBO_12",
                "structured_caption": {"description": "generation"},
            },
        ),
        lambda: gateway.post(
            "/v1/image-edits",
            files=source,
            data={"model": "longcat-image-edit", "prompt": "edit", "seed": "9"},
        ),
        lambda: gateway.post("/v1/background-removal?model=birefnet-hr-matting", files=source),
        lambda: gateway.post("/v1/models/unload"),
    )
    for request in busy_requests:
        response = request()
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "image_capacity_busy"
    assert len(dispatches) == 1
    assert not boundary.calls

    boundary.release.set()
    held.join()
    response = held_result[0]
    assert getattr(response, "status_code") == 200
    assert max_active_models == 1

    generation_response = gateway.post(
        "/v1/generations",
        json={
            "width": 256,
            "height": 256,
            "seed": 7,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "generation"},
        },
    )
    assert generation_response.status_code == 200
    assert dispatches[-1][0:2] == ("generation", "/internal/generate")
    assert boundary.calls[-1] == {
        "width": 256,
        "height": 256,
        "seed": 7,
        "sampler_preset": "V4_TURBO_12",
        "structured_caption": {"description": "generation"},
        "magic_prompt": False,
        "model": "ideogram-4-nf4",
    }

    edit_response = gateway.post(
        "/v1/image-edits",
        files=source,
        data={
            "model": "longcat-image-edit",
            "prompt": "edit",
            "negative_prompt": "none",
            "seed": "9",
        },
    )
    background_response = gateway.post(
        "/v1/background-removal?model=birefnet-hr-matting", files=source
    )
    unload_response = gateway.post("/v1/models/unload")
    assert (
        edit_response.status_code
        == background_response.status_code
        == unload_response.status_code
        == 200
    )
    assert [item[1].split("?")[0] for item in dispatches] == [
        "/internal/upscale",
        "/internal/generate",
        "/internal/image-edit",
        "/internal/background-removal",
        "/internal/unload",
        "/internal/unload",
        "/internal/unload",
    ]
    assert b'name="file"' in dispatches[2][2]
    assert png() in dispatches[2][2]
    assert parse_qs(urlsplit(dispatches[2][1]).query) == {
        "model": ["longcat-image-edit"],
        "prompt": ["edit"],
        "negative_prompt": ["none"],
        "seed": ["9"],
    }
    assert parse_qs(urlsplit(dispatches[3][1]).query) == {
        "model": ["birefnet-hr-matting"],
        "alpha_blur": ["0.0"],
        "alpha_erode": ["0"],
        "alpha_dilate": ["0"],
        "alpha_threshold": ["0"],
        "birefnet_inference_size": ["2048"],
        "birefnet_foreground_refinement": ["False"],
        "model_input_size": ["1024"],
        "sam2_guidance": ["False"],
        "sam2_model": ["sam2.1-hiera-large"],
        "sam2_mask_threshold": ["0.5"],
        "sam2_prompt_alpha_threshold": ["128"],
        "sam2_interior_erode": ["4"],
        "sam2_boundary_dilate": ["8"],
        "boundary_alpha_gamma": ["0.6"],
    }
    assert models.child_alive is False
    assert models.loaded_model is None

    before_connect_failure = len(dispatches)

    def connect_failure(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "loaded": False,
                    "weightsAvailable": True,
                    "device": "cpu-test",
                    "models": {
                        "ideogram-4-nf4": {"weightsAvailable": True},
                        "longcat-image-edit": {"weightsAvailable": True},
                        "longcat-image-edit-turbo": {"weightsAvailable": True},
                    },
                },
                request=request,
            )
        raise httpx.ConnectError("fixture connect failure", request=request)

    workers.client = httpx.Client(transport=httpx.MockTransport(connect_failure), trust_env=False)
    retryable = gateway.post(
        "/v1/generations",
        json={
            "width": 256,
            "height": 256,
            "seed": 3,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "connect"},
        },
    )
    assert retryable.status_code == 503
    assert retryable.json()["error"]["code"] == "worker_unavailable"
    assert len(dispatches) == before_connect_failure

    boundary.failure = RuntimeError("fixture output failure after handler entry")
    workers.client = HttpWorkerClient(
        "http://upscale",
        "http://background",
        1,
        1_000_000,
        httpx.MockTransport(transport),
        "http://generation",
    ).client
    ambiguous = gateway.post(
        "/v1/generations",
        json={
            "width": 256,
            "height": 256,
            "seed": 4,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "post-entry"},
        },
    )
    assert ambiguous.status_code == 502
    assert ambiguous.json()["error"]["code"] == "worker_execution_unknown"
    assert boundary.calls[-1]["structured_caption"] == {"description": "post-entry"}
    boundary.failure = None

    workers.client = HttpWorkerClient(
        "http://upscale",
        "http://background",
        1,
        1_000_000,
        httpx.MockTransport(transport),
        "http://generation",
    ).client
    recovered = gateway.post(
        "/v1/generations",
        json={
            "width": 256,
            "height": 256,
            "seed": 5,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "recovered"},
        },
    )
    assert recovered.status_code == 200

    boundary.failure = RuntimeError("fixture model failure after handler entry")
    ordinary_failure = gateway.post(
        "/v1/image-edits",
        files=source,
        data={
            "model": "longcat-image-edit",
            "prompt": "ordinary-failure",
            "seed": "6",
        },
    )
    assert ordinary_failure.status_code == 502
    assert coordinator.status() == {"ready": True, "active": 0, "capacity": 1}
    boundary.failure = None
    workers.client = HttpWorkerClient(
        "http://upscale",
        "http://background",
        1,
        1_000_000,
        httpx.MockTransport(transport),
        "http://generation",
    ).client
    assert (
        gateway.post(
            "/v1/generations",
            json={
                "width": 256,
                "height": 256,
                "seed": 8,
                "sampler_preset": "V4_TURBO_12",
                "structured_caption": {"description": "after-ordinary-failure"},
            },
        ).status_code
        == 200
    )
