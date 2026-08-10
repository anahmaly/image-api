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
    LONGCAT_EDIT_TURBO_REVISION,
    MAX_SNAPSHOT_JSON_BYTES,
    Settings,
    ideogram_weights_available,
    longcat_weights_available,
)
from image_api.workers import FakeWorkerClient, HttpWorkerClient
from image_api_workers import upscale


@pytest.mark.parametrize(
    ("path", "settings_key"),
    (("/v1/image-edits", "max_request_bytes"), ("/v1/upscale", "processing_max_request_bytes")),
)
def test_gateway_rejects_declared_and_streamed_request_bodies_before_multipart(
    tmp_path, path: str, settings_key: str
) -> None:
    settings = Settings.for_tests(tmp_path, **{settings_key: 64})
    app = create_app(settings=settings)
    client = TestClient(app)

    declared = client.post(
        path,
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

        await RequestBodyLimitMiddleware(downstream, 1_000, {path: 64})(
            {"type": "http", "method": "POST", "path": path, "headers": []}, receive, send
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


@pytest.mark.parametrize(
    ("path", "fields", "settings_key"),
    (
        (
            "/v1/image-edits",
            b'Content-Disposition: form-data; name="model"\r\n\r\nlongcat-image-edit\r\n'
            b'--test-boundary\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nedit\r\n'
            b'--test-boundary\r\nContent-Disposition: form-data; name="seed"\r\n\r\n9\r\n',
            "max_request_bytes",
        ),
        (
            "/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=0",
            b"",
            "processing_max_request_bytes",
        ),
    ),
)
def test_multipart_file_ceiling_and_request_ceiling_are_distinct(
    tmp_path, path: str, fields: bytes, settings_key: str
) -> None:
    source = png()
    boundary = b"test-boundary"
    body = b"--" + boundary + b"\r\n" + fields
    if fields:
        body += b"--" + boundary + b"\r\n"
    body += (
        b'Content-Disposition: form-data; name="file"; filename="input.png"\r\n'
        b"Content-Type: image/png\r\n\r\n" + source + b"\r\n--" + boundary + b"--\r\n"
    )
    settings = Settings.for_tests(
        tmp_path,
        max_upload_bytes=len(source),
        processing_max_upload_bytes=len(source),
        **{settings_key: len(body)},
    )
    client = TestClient(create_app(settings=settings, workers=FakeWorkerClient()))
    headers = {"content-type": f"multipart/form-data; boundary={boundary.decode()}"}
    accepted = client.post(path, content=body, headers=headers)
    rejected = client.post(path, content=body + b"x", headers=headers)
    assert accepted.status_code == 200
    assert rejected.status_code == 413


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


def _longcat_snapshot(root, revision: str, *, sharded: bool = False):
    root.mkdir(parents=True)
    (root / ".image-api-revision").write_text(revision)
    for name in (
        "config.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/preprocessor_config.json",
        "text_processor/chat_template.json",
        "text_processor/config.json",
        "text_processor/preprocessor_config.json",
        "text_processor/special_tokens_map.json",
        "text_processor/tokenizer.json",
        "text_processor/tokenizer_config.json",
        "text_processor/vocab.json",
        "tokenizer/chat_template.json",
        "tokenizer/config.json",
        "tokenizer/preprocessor_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "vae/config.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for name in ("text_processor/merges.txt", "tokenizer/merges.txt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("merge")
    for component, filename in (
        ("text_encoder", "model.safetensors"),
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
    ):
        directory = root / component
        if sharded:
            shard = "weights-00001.safetensors"
            (directory / f"{filename}.index.json").write_text(
                json.dumps({"weight_map": {"layer": shard}})
            )
            (directory / shard).write_bytes(b"weights")
        else:
            (directory / filename).write_bytes(b"weights")
    return root


def test_snapshot_validation_requires_direct_or_safe_complete_shards(tmp_path) -> None:
    snapshot = _ideogram_snapshot(tmp_path, sharded=True)
    assert ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")
    index = snapshot / "transformer/diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer": "../escape.safetensors"}}))
    assert not ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")


@pytest.mark.parametrize("sharded", (False, True))
def test_complete_ideogram_snapshot_is_ready(tmp_path, sharded: bool) -> None:
    _ideogram_snapshot(tmp_path, sharded=sharded)
    assert ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")


@pytest.mark.parametrize("revision", (LONGCAT_EDIT_REVISION, LONGCAT_EDIT_TURBO_REVISION))
@pytest.mark.parametrize("sharded", (False, True))
def test_complete_longcat_snapshot_is_ready(tmp_path, revision: str, sharded: bool) -> None:
    root = _longcat_snapshot(tmp_path / "longcat", revision, sharded=sharded)
    assert longcat_weights_available(root, revision)


@pytest.mark.parametrize("snapshot", ("ideogram", "longcat"))
@pytest.mark.parametrize("failure", ("malformed", "oversized", "missing", "unsafe-index", "marker"))
def test_snapshot_validation_rejects_malformed_or_unsafe_loader_inputs(
    tmp_path, snapshot: str, failure: str
) -> None:
    if snapshot == "ideogram":
        root = _ideogram_snapshot(tmp_path, sharded=True)
        marker = tmp_path / "hub/models--ideogram-ai--ideogram-4-nf4/refs/main"
        json_file = root / "vae/config.json"
        index = root / "transformer/diffusion_pytorch_model.safetensors.index.json"
        assert ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")
    else:
        root = _longcat_snapshot(tmp_path / "longcat", LONGCAT_EDIT_REVISION, sharded=True)
        marker = root / ".image-api-revision"
        json_file = root / "config.json"
        index = root / "transformer/diffusion_pytorch_model.safetensors.index.json"
        assert longcat_weights_available(root, LONGCAT_EDIT_REVISION)
    if failure == "malformed":
        json_file.write_text("{")
    elif failure == "oversized":
        json_file.write_bytes(b"{" + b"x" * MAX_SNAPSHOT_JSON_BYTES)
    elif failure == "missing":
        json_file.unlink()
    elif failure == "unsafe-index":
        index.write_text(json.dumps({"weight_map": {"layer": "../escape.safetensors"}}))
    else:
        marker.write_text("a" * 66)
    if snapshot == "ideogram":
        assert not ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")
    else:
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
