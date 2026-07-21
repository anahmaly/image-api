from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers import png
from image_api.app import create_app
from image_api.config import Settings, ideogram_weights_available
from image_api.store import TaskStore
from image_api.workers import HttpWorkerClient, WorkerUnavailable


class BrokenWorkers:
    model_invocations = 0
    model_loads = 0

    def health(self):
        return {
            "upscale": {
                "ready": False,
                "loaded": False,
                "device": "/dev/private-gpu",
                "privatePath": "/models/secret",
            }
        }

    def upscale(self, *args, **kwargs):
        raise WorkerUnavailable("http://private-worker:9001 exploded /models/secret")

    def background(self, *args, **kwargs):
        raise WorkerUnavailable("socket detail")


class TrackingStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.yielded = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk


@pytest.mark.parametrize("content_length", [None, "1000"])
def test_worker_stream_cap_aborts_and_gateway_returns_one_safe_failure(
    tmp_path, content_length
) -> None:
    stream = TrackingStream([b"1234", b"5678", b"must-not-be-read"])

    def handler(_request: httpx.Request) -> httpx.Response:
        headers = {} if content_length is None else {"content-length": content_length}
        return httpx.Response(200, headers=headers, stream=stream)

    workers = HttpWorkerClient(
        "http://upscale-worker",
        "http://background-worker",
        timeout_seconds=1,
        max_output_bytes=7,
        transport=httpx.MockTransport(handler),
    )
    settings = Settings.for_tests(tmp_path)
    client = TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=workers)
    )
    response = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
        files={"file": ("x.png", png(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "worker_unavailable",
            "message": "Image capability is temporarily unavailable",
        }
    }
    assert "1000" not in response.text
    assert stream.yielded == (2 if content_length is None else 0)


def test_worker_failure_is_typed_and_safe(tmp_path) -> None:
    settings = Settings.for_tests(tmp_path)
    client = TestClient(
        create_app(
            settings=settings, store=TaskStore(settings.database_path), workers=BrokenWorkers()
        )
    )
    response = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512",
        files={"file": ("x.png", png(), "image/png")},
    )
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "worker_unavailable",
            "message": "Image capability is temporarily unavailable",
        }
    }
    assert "private-worker" not in response.text


def test_missing_generation_runtime_is_honestly_disabled(tmp_path) -> None:
    settings = Settings.for_tests(
        tmp_path,
        ideogram_weights_path=tmp_path / "missing",
        cuda_available=False,
        generation_test_mode=False,
    )
    client = TestClient(
        create_app(
            settings=settings, store=TaskStore(settings.database_path), workers=BrokenWorkers()
        )
    )
    health_response = client.get("/health")
    assert "/models/" not in health_response.text
    assert "/dev/" not in health_response.text
    health = health_response.json()
    generation = health["capabilities"]["generation"]
    assert generation["ready"] is False
    assert generation["reason"] in {"weights_unavailable", "cuda_unavailable"}


IDEOGRAM_REPOSITORY_ID = "ideogram-ai/ideogram-4-nf4"
OFFICIAL_SNAPSHOT = "f664347839e0a87bc495f5c9483cc0014b8e344e"
DIFFUSION_COMPONENTS = ("transformer", "unconditional_transformer")


def create_ideogram_snapshot(tmp_path):
    repository = tmp_path / "hub" / "models--ideogram-ai--ideogram-4-nf4"
    reference = repository / "refs" / "main"
    reference.parent.mkdir(parents=True)
    reference.write_text(OFFICIAL_SNAPSHOT)
    snapshot = repository / "snapshots" / OFFICIAL_SNAPSHOT
    required = (
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
    )
    for relative in required:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return snapshot


def add_direct_diffusion_weights(snapshot, component):
    weights = snapshot / component / "diffusion_pytorch_model.safetensors"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")


def add_sharded_diffusion_weights(snapshot, component):
    directory = snapshot / component
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    shard = directory / "weights-00001.safetensors"
    directory.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"weight_map": {"layer": shard.name}}))
    shard.write_bytes(b"weights")
    return index, shard


def test_official_unsharded_ideogram_snapshot_is_available(tmp_path) -> None:
    snapshot = create_ideogram_snapshot(tmp_path)
    for component in DIFFUSION_COMPONENTS:
        add_direct_diffusion_weights(snapshot, component)

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is True


def test_both_sharded_diffusion_components_are_available(tmp_path) -> None:
    snapshot = create_ideogram_snapshot(tmp_path)
    for component in DIFFUSION_COMPONENTS:
        add_sharded_diffusion_weights(snapshot, component)

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is True


def test_direct_and_sharded_diffusion_components_are_independently_available(tmp_path) -> None:
    snapshot = create_ideogram_snapshot(tmp_path)
    add_direct_diffusion_weights(snapshot, "transformer")
    add_sharded_diffusion_weights(snapshot, "unconditional_transformer")

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is True


def test_missing_direct_diffusion_weights_are_unavailable(tmp_path) -> None:
    snapshot = create_ideogram_snapshot(tmp_path)
    add_direct_diffusion_weights(snapshot, "transformer")

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is False


@pytest.mark.parametrize(
    "invalid_index", ["malformed", "empty", "oversized", "absolute", "traversal", "missing"]
)
def test_invalid_diffusion_indexes_are_unavailable(tmp_path, invalid_index) -> None:
    snapshot = create_ideogram_snapshot(tmp_path)
    add_direct_diffusion_weights(snapshot, "transformer")
    index, shard = add_sharded_diffusion_weights(snapshot, "unconditional_transformer")

    if invalid_index == "malformed":
        index.write_text("{")
    elif invalid_index == "empty":
        index.write_text(json.dumps({"weight_map": {}}))
    elif invalid_index == "oversized":
        index.write_bytes(b" " * 5_000_001)
    elif invalid_index == "absolute":
        index.write_text(json.dumps({"weight_map": {"layer": "/weights.safetensors"}}))
    elif invalid_index == "traversal":
        index.write_text(json.dumps({"weight_map": {"layer": "../weights.safetensors"}}))
    else:
        shard.unlink()

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is False


def test_text_encoder_sharded_weights_are_available(tmp_path) -> None:
    snapshot = create_ideogram_snapshot(tmp_path)
    for component in DIFFUSION_COMPONENTS:
        add_direct_diffusion_weights(snapshot, component)
    (snapshot / "text_encoder/model.safetensors").unlink()
    index = snapshot / "text_encoder/model.safetensors.index.json"
    shard = snapshot / "text_encoder/model-00001-of-00001.safetensors"
    index.write_text(json.dumps({"weight_map": {"encoder": shard.name}}))
    shard.write_bytes(b"weights")

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is True


def test_missing_text_encoder_weights_are_unavailable(tmp_path) -> None:
    snapshot = create_ideogram_snapshot(tmp_path)
    for component in DIFFUSION_COMPONENTS:
        add_direct_diffusion_weights(snapshot, component)
    (snapshot / "text_encoder/model.safetensors").unlink()

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is False


@pytest.mark.parametrize("reference", ["", "not-a-sha", "a" * 65])
def test_invalid_main_reference_is_unavailable(tmp_path, reference) -> None:
    create_ideogram_snapshot(tmp_path)
    reference_path = tmp_path / "hub/models--ideogram-ai--ideogram-4-nf4/refs/main"
    reference_path.write_text(reference)

    assert ideogram_weights_available(tmp_path, IDEOGRAM_REPOSITORY_ID) is False
