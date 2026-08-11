from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers import png
from spawn_adapters import build_adapters, settings as spawn_settings
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
from image_api_workers.generation_models import GenerationModels
from image_api_workers.generation_worker import create_worker_app


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


def _longcat_snapshot(root, revision: str, *, sharded: bool = False, merge_size: int | None = None):
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
        if merge_size is None:
            path.write_text("merge")
        else:
            path.write_text("merge")
            with path.open("r+b") as merge_file:
                merge_file.truncate(merge_size)
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


def _write_json_of_size(path, size: int) -> None:
    prefix = b'{"tokenizer":"'
    suffix = b'"}'
    path.write_bytes(prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix)


def test_production_composed_physical_model_layout_admits_all_configured_models(
    monkeypatch, tmp_path
) -> None:
    models_root = tmp_path / "models"
    _ideogram_snapshot(models_root / "ideogram-4-nf4")
    standard = _longcat_snapshot(
        models_root / "longcat-image-edit",
        LONGCAT_EDIT_REVISION,
        merge_size=1_671_839,
    )
    turbo = _longcat_snapshot(
        models_root / "longcat-image-edit-turbo",
        LONGCAT_EDIT_TURBO_REVISION,
        merge_size=1_671_839,
    )
    settings = Settings.for_tests(
        tmp_path,
        generation_test_mode=False,
        ideogram_weights_path=models_root / "ideogram-4-nf4",
        longcat_edit_weights_path=standard,
        longcat_edit_turbo_weights_path=turbo,
    )

    class Adapter:
        def __call__(self, request: dict[str, object]) -> bytes:
            return b""

        def unload(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )
    health = (
        TestClient(
            create_worker_app(
                GenerationModels(spawn_settings(), adapter_factory=build_adapters), settings
            )
        )
        .get("/health")
        .json()
    )

    assert ideogram_weights_available(models_root / "ideogram-4-nf4", "ideogram-ai/ideogram-4-nf4")
    assert longcat_weights_available(standard, LONGCAT_EDIT_REVISION)
    assert longcat_weights_available(turbo, LONGCAT_EDIT_TURBO_REVISION)
    assert health["ready"] is True
    assert health["models"] == {
        "ideogram-4-nf4": {"weightsAvailable": True, "loaded": False},
        "longcat-image-edit": {"weightsAvailable": True, "loaded": False},
        "longcat-image-edit-turbo": {"weightsAvailable": True, "loaded": False},
    }


def test_official_tokenizer_json_sizes_reach_generation_readiness(monkeypatch, tmp_path) -> None:
    settings = Settings.for_tests(tmp_path, generation_test_mode=False)
    ideogram = _ideogram_snapshot(settings.ideogram_weights_path)
    standard = _longcat_snapshot(settings.longcat_edit_weights_path, LONGCAT_EDIT_REVISION)
    turbo = _longcat_snapshot(settings.longcat_edit_turbo_weights_path, LONGCAT_EDIT_TURBO_REVISION)
    _write_json_of_size(ideogram / "tokenizer/tokenizer.json", 11_422_650)
    for snapshot in (standard, turbo):
        _write_json_of_size(snapshot / "text_processor/tokenizer.json", 7_031_645)
        _write_json_of_size(snapshot / "tokenizer/tokenizer.json", 7_031_645)

    class Adapter:
        def __call__(self, request: dict[str, object]) -> bytes:
            return b""

        def unload(self) -> None:
            pass

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )
    health = (
        TestClient(
            create_worker_app(
                GenerationModels(spawn_settings(), adapter_factory=build_adapters), settings
            )
        )
        .get("/health")
        .json()
    )

    assert health["ready"] is True
    assert health["weightsAvailable"] is True
    assert health["models"] == {
        "ideogram-4-nf4": {"weightsAvailable": True, "loaded": False},
        "longcat-image-edit": {"weightsAvailable": True, "loaded": False},
        "longcat-image-edit-turbo": {"weightsAvailable": True, "loaded": False},
    }


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
@pytest.mark.parametrize(
    "failure",
    ("malformed", "oversized", "missing", "empty", "unsafe-index", "missing-shard", "marker"),
)
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
    elif failure == "empty":
        json_file.write_bytes(b"")
    elif failure == "unsafe-index":
        index.write_text(json.dumps({"weight_map": {"layer": "../escape.safetensors"}}))
    elif failure == "missing-shard":
        (index.parent / "weights-00001.safetensors").unlink()
    else:
        marker.write_text("a" * 66)
    if snapshot == "ideogram":
        assert not ideogram_weights_available(tmp_path, "ideogram-ai/ideogram-4-nf4")
    else:
        assert not longcat_weights_available(root, LONGCAT_EDIT_REVISION)


@pytest.mark.parametrize(
    ("unavailable_model", "failure"),
    (
        (None, None),
        ("ideogram-4-nf4", "missing"),
        ("ideogram-4-nf4", "malformed"),
        ("longcat-image-edit", "missing"),
        ("longcat-image-edit", "malformed"),
        ("longcat-image-edit-turbo", "missing"),
        ("longcat-image-edit-turbo", "malformed"),
    ),
)
def test_worker_readiness_matrix_reaches_gateway_and_blocks_unavailable_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    unavailable_model: str | None,
    failure: str | None,
) -> None:
    """The actual worker health matrix is the gateway admission authority."""

    settings = Settings.for_tests(tmp_path, generation_test_mode=False)
    ideogram = _ideogram_snapshot(settings.ideogram_weights_path)
    standard = _longcat_snapshot(settings.longcat_edit_weights_path, LONGCAT_EDIT_REVISION)
    turbo = _longcat_snapshot(settings.longcat_edit_turbo_weights_path, LONGCAT_EDIT_TURBO_REVISION)
    snapshots = {
        "ideogram-4-nf4": ideogram / "vae/config.json",
        "longcat-image-edit": standard / "config.json",
        "longcat-image-edit-turbo": turbo / "config.json",
    }
    if unavailable_model is not None:
        target = snapshots[unavailable_model]
        if failure == "missing":
            target.unlink()
        else:
            target.write_text("{")

    class IdeogramBoundary:
        def __call__(self, request: dict[str, object]) -> bytes:
            model = request["model"]
            assert isinstance(model, str)
            width = request["width"]
            height = request["height"]
            assert type(width) is int and type(height) is int
            return png(size=(width, height))

        def unload(self) -> None:
            pass

    class LongCatBoundary:
        loaded_model: str | None = None

        def __call__(self, request: dict[str, object]) -> bytes:
            model = request["model"]
            assert isinstance(model, str)
            self.loaded_model = model
            source = request["source_image_bytes"]
            assert isinstance(source, bytes)
            return source

        def unload(self) -> None:
            self.loaded_model = None

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )
    lifecycle: list[tuple[str, str, int]] = []
    models = GenerationModels(
        spawn_settings(),
        adapter_factory=build_adapters,
        lifecycle_observer=lambda event, model, live_children: lifecycle.append(
            (event, model, live_children)
        ),
    )
    generation = TestClient(create_worker_app(models, settings))
    dispatches: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.host == "generation":
            if request.url.path != "/health":
                dispatches.append(request.url.path)
            response = generation.request(
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
        return httpx.Response(
            200,
            json={"ready": True, "loaded": False, "device": "cpu-test"},
            request=request,
        )

    workers = HttpWorkerClient(
        "http://upscale",
        "http://background",
        1,
        1_000_000,
        httpx.MockTransport(transport),
        "http://generation",
    )
    gateway = TestClient(create_app(settings=settings, workers=workers))
    health = gateway.get("/health").json()

    def dispatch_selected(model: str, *, seed: int) -> str:
        if model == "ideogram-4-nf4":
            response = gateway.post(
                "/v1/generations",
                json={
                    "width": 256,
                    "height": 256,
                    "seed": seed,
                    "sampler_preset": "V4_TURBO_12",
                    "structured_caption": {"description": "generation"},
                },
            )
            expected_dispatch = "/internal/generate"
        else:
            response = gateway.post(
                "/v1/image-edits",
                files={"file": ("input.png", png(), "image/png")},
                data={"model": model, "prompt": "edit", "seed": str(seed)},
            )
            expected_dispatch = "/internal/image-edit"
        assert response.status_code == 200
        assert generation.get("/health").json()["loadedModel"] == model
        return expected_dispatch

    if unavailable_model is None:
        assert health["status"] == "ok"
        assert health["capabilities"]["generation"]["ready"] is True
        expected_dispatches = [
            dispatch_selected(model, seed=index)
            for index, model in enumerate(
                ("ideogram-4-nf4", "longcat-image-edit", "longcat-image-edit-turbo"), start=1
            )
        ]
        assert dispatches == expected_dispatches
        assert lifecycle == [
            ("spawn", "ideogram-4-nf4", 1),
            ("load", "ideogram-4-nf4", 1),
            ("exit", "ideogram-4-nf4", 0),
            ("reap", "ideogram-4-nf4", 0),
            ("spawn", "longcat-image-edit", 1),
            ("load", "longcat-image-edit", 1),
            ("exit", "longcat-image-edit", 0),
            ("reap", "longcat-image-edit", 0),
            ("spawn", "longcat-image-edit-turbo", 1),
            ("load", "longcat-image-edit-turbo", 1),
        ]
        assert max(live_children for _, _, live_children in lifecycle) == 1
        models.unload()
        return

    assert health["status"] == "degraded"
    generation_health = health["capabilities"]["generation"]
    assert generation_health["ready"] is False
    assert generation_health["models"][unavailable_model]["weightsAvailable"] is False
    available_models = tuple(
        model
        for model in ("ideogram-4-nf4", "longcat-image-edit", "longcat-image-edit-turbo")
        if model != unavailable_model
    )
    expected_dispatches = [
        dispatch_selected(model, seed=index)
        for index, model in enumerate(available_models, start=1)
    ]
    assert dispatches == expected_dispatches
    dispatch_count_before_rejection = len(dispatches)
    if unavailable_model == "ideogram-4-nf4":
        response = gateway.post(
            "/v1/generations",
            json={
                "width": 256,
                "height": 256,
                "seed": 1,
                "sampler_preset": "V4_TURBO_12",
                "structured_caption": {"description": "generation"},
            },
        )
    else:
        response = gateway.post(
            "/v1/image-edits",
            files={"file": ("input.png", png(), "image/png")},
            data={"model": unavailable_model, "prompt": "edit", "seed": "1"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "worker_unavailable"
    assert len(dispatches) == dispatch_count_before_rejection
    models.unload()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ready", "true"),
        ("loaded", 1),
        ("weightsAvailable", ["available"]),
        ("models.ideogram-4-nf4.weightsAvailable", {}),
        ("models.ideogram-4-nf4.loaded", "false"),
        ("models.longcat-image-edit.weightsAvailable", 1),
        ("models.longcat-image-edit.loaded", ["loaded"]),
        ("models.longcat-image-edit-turbo.weightsAvailable", "true"),
        ("models.longcat-image-edit-turbo.loaded", {}),
    ),
)
def test_http_worker_health_accepts_only_literal_json_booleans(field: str, value: object) -> None:
    body: dict[str, object] = {
        "ready": True,
        "loaded": True,
        "weightsAvailable": True,
        "device": "cpu-test",
        "models": {
            "ideogram-4-nf4": {"weightsAvailable": True, "loaded": True},
            "longcat-image-edit": {"weightsAvailable": True, "loaded": True},
            "longcat-image-edit-turbo": {"weightsAvailable": True, "loaded": True},
        },
    }
    target: dict[str, object] = body
    *parents, leaf = field.split(".")
    for parent in parents:
        target = cast(dict[str, object], target[parent])
    target[leaf] = value

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, request=request)

    health = HttpWorkerClient(
        "http://upscale", "http://background", 1, 1_000_000, httpx.MockTransport(transport)
    ).health()["generation"]
    expected = field.split(".")[-1]
    if field.startswith("models."):
        model = field.split(".")[1]
        models = cast(dict[str, dict[str, object]], health["models"])
        assert models[model][expected] is False
    else:
        assert health[expected] is False


@pytest.mark.parametrize(
    ("schema_case", "selected_model"),
    (
        ("top-ready", "ideogram-4-nf4"),
        ("top-weights", "ideogram-4-nf4"),
        ("ideogram-weights", "ideogram-4-nf4"),
        ("standard-weights", "longcat-image-edit"),
        ("turbo-weights", "longcat-image-edit-turbo"),
        ("missing-standard", "longcat-image-edit"),
        ("unknown-extra", "ideogram-4-nf4"),
    ),
)
def test_malformed_worker_health_json_degrades_gateway_and_blocks_dispatch(
    tmp_path, schema_case: str, selected_model: str
) -> None:
    health: dict[str, object] = {
        "ready": True,
        "loaded": False,
        "weightsAvailable": True,
        "device": "cpu-test",
        "models": {
            "ideogram-4-nf4": {"weightsAvailable": True, "loaded": False},
            "longcat-image-edit": {"weightsAvailable": True, "loaded": False},
            "longcat-image-edit-turbo": {"weightsAvailable": True, "loaded": False},
        },
    }
    models = cast(dict[str, dict[str, object]], health["models"])
    if schema_case == "top-ready":
        health["ready"] = "true"
    elif schema_case == "top-weights":
        health["weightsAvailable"] = ["available"]
    elif schema_case == "ideogram-weights":
        models["ideogram-4-nf4"]["weightsAvailable"] = "true"
    elif schema_case == "standard-weights":
        models["longcat-image-edit"]["weightsAvailable"] = 1
    elif schema_case == "turbo-weights":
        models["longcat-image-edit-turbo"]["weightsAvailable"] = {}
    elif schema_case == "missing-standard":
        del models["longcat-image-edit"]
    else:
        del models["ideogram-4-nf4"]
        models["unknown-model"] = {"weightsAvailable": True, "loaded": True}
    dispatches: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.host == "generation" and request.url.path == "/health":
            return httpx.Response(200, json=health, request=request)
        if request.url.host == "generation":
            dispatches.append(request.url.path)
        return httpx.Response(
            200,
            json={"ready": True, "loaded": False, "device": "cpu-test"},
            request=request,
        )

    workers = HttpWorkerClient(
        "http://upscale",
        "http://background",
        1,
        1_000_000,
        httpx.MockTransport(transport),
        "http://generation",
    )
    gateway = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=workers))
    assert gateway.get("/health").json()["status"] == "degraded"
    if selected_model == "ideogram-4-nf4":
        response = gateway.post(
            "/v1/generations",
            json={
                "width": 256,
                "height": 256,
                "seed": 1,
                "sampler_preset": "V4_TURBO_12",
                "structured_caption": {"description": "generation"},
            },
        )
    else:
        response = gateway.post(
            "/v1/image-edits",
            files={"file": ("input.png", png(), "image/png")},
            data={"model": selected_model, "prompt": "edit", "seed": "1"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "worker_unavailable"
    assert dispatches == []


def test_http_worker_phases_keep_only_connection_refusal_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request)

    client = HttpWorkerClient(
        "http://upscale", "http://background", 1, 1_000_000, httpx.MockTransport(handler)
    )
    with pytest.raises(Exception) as failed:
        client.upscale(png(), model="RealESRGAN_x4plus", outscale=2, tile=0)
    assert failed.value.__class__.__name__ == "WorkerExecutionFailed"
