from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers import png
from image_api.app import RequestBodyLimitMiddleware, create_app
from image_api.config import (
    LONGCAT_EDIT_REVISION,
    Settings,
    ideogram_weights_available,
    longcat_weights_available,
)
from image_api.workers import HttpWorkerClient
from image_api_workers import upscale


def test_gateway_rejects_declared_and_streamed_processing_bodies_before_multipart(tmp_path) -> None:
    settings = Settings.for_tests(tmp_path, processing_max_upload_bytes=64)
    app = create_app(settings=settings)
    client = TestClient(app)

    declared = client.post(
        "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=0",
        content=b"x" * 65,
        headers={"content-length": "65", "content-type": "multipart/form-data; boundary=x"},
    )
    malformed = client.post("/unknown", content=b"x", headers={"content-length": "not-a-size"})

    async def streamed_body() -> list[dict[str, object]]:
        received = iter((b"x" * 32, b"x" * 33))
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            try:
                return {"type": "http.request", "body": next(received), "more_body": True}
            except StopIteration:
                return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def downstream(scope, receive, send) -> None:
            while (await receive()).get("more_body"):
                pass

        await RequestBodyLimitMiddleware(downstream, 64, {})(
            {"type": "http", "method": "POST", "path": "/v1/upscale", "headers": []}, receive, send
        )
        return sent

    streamed = asyncio.run(streamed_body())
    assert declared.status_code == 413
    assert declared.json()["error"]["code"] == "request_too_large"
    assert (
        next(message for message in streamed if message["type"] == "http.response.start")["status"]
        == 413
    )
    assert malformed.status_code == 400


def test_realesrgan_native_limits_and_forced_large_tiling(monkeypatch) -> None:
    assert upscale._effective_tile(0, 4096, 4096) == 512
    assert upscale._effective_tile(0, 8192, 8192) == 512
    assert upscale._effective_tile(768, 4096, 4096) == 768
    monkeypatch.setenv("IMAGE_API_PROCESSING_MAX_NATIVE_PIXELS", "5")
    with pytest.raises(Exception, match="native processing"):
        upscale._run(png(size=(2, 2)), "RealESRGAN_x4plus", 1, 0)


def _ideogram_snapshot(root, *, sharded: bool = False):
    snapshot_id = "a" * 40
    snapshot = root / "hub/models--ideogram-ai--ideogram-4-nf4/snapshots" / snapshot_id
    reference = snapshot.parent.parent / "refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text(snapshot_id)
    for name in (
        "vae/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
    ):
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for component, filename in (
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("unconditional_transformer", "diffusion_pytorch_model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
        ("text_encoder", "model.safetensors"),
    ):
        directory = snapshot / component
        directory.mkdir(parents=True, exist_ok=True)
        if sharded:
            shard = "weights-00001.safetensors"
            (directory / f"{filename}.index.json").write_text(
                json.dumps({"weight_map": {"layer": shard}})
            )
            (directory / shard).write_bytes(b"weights")
        else:
            (directory / filename).write_bytes(b"weights")
    return snapshot


def test_snapshot_validation_requires_direct_or_safe_complete_shards(tmp_path) -> None:
    snapshot = _ideogram_snapshot(tmp_path, sharded=True)
    assert ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")
    index = snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer": "../escape.safetensors"}}))
    assert not ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")


def test_longcat_snapshot_requires_loaded_pipeline_components(tmp_path) -> None:
    root = tmp_path / "longcat"
    root.mkdir()
    (root / ".image-api-revision").write_text(LONGCAT_EDIT_REVISION)
    assert not longcat_weights_available(root, LONGCAT_EDIT_REVISION)


def test_http_worker_phases_keep_only_connection_refusal_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request)

    client = HttpWorkerClient(
        "http://upscale", "http://background", 1, 1_000_000, httpx.MockTransport(handler)
    )
    with pytest.raises(Exception) as failed:
        client.upscale(png(), model="RealESRGAN_x4plus", outscale=2, tile=0)
    assert failed.value.__class__.__name__ == "WorkerExecutionFailed"
