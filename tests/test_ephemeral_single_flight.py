from __future__ import annotations

from fastapi.testclient import TestClient

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.coordinator import SingleFlightCoordinator
from image_api.workers import FakeWorkerClient, WorkerUnavailable


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
    assert client.post("/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=0", files=source).headers["content-type"].startswith("image/png")
    assert client.post("/v1/background-removal?model=birefnet-hr-matting", files=source).headers["content-type"].startswith("image/png")
    assert client.post("/v1/generations", json={"width": 256, "height": 256, "seed": 1, "sampler_preset": "V4_TURBO_12", "structured_caption": {"description": "bee"}}).headers["content-type"].startswith("image/png")
    assert worker.model_invocations == 3


def test_worker_unavailability_is_retryable_without_replay(tmp_path) -> None:
    class UnavailableWorker(FakeWorkerClient):
        def generation(self, request: dict[str, object]) -> bytes:
            self.model_invocations += 1
            raise WorkerUnavailable("not ready")

    worker = UnavailableWorker()
    client = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=worker))
    response = client.post("/v1/generations", json={"width": 256, "height": 256, "seed": 1, "sampler_preset": "V4_TURBO_12", "structured_caption": {"description": "bee"}})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "worker_unavailable"
    assert worker.model_invocations == 1
