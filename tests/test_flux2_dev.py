from __future__ import annotations

import json
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from flux2_dev_fakes import HeavyBoundaries, inline_processes, snapshot
from helpers import png
from image_api.app import create_app
from image_api.config import (
    FLUX_2_DEV_MANIFEST,
    FLUX_2_DEV_REVISION,
    Settings,
    flux_2_dev_weights_available,
)
from image_api.coordinator import SingleFlightCoordinator
from image_api.workers import HttpWorkerClient, PeerEvictor
from image_api_workers.flux2_dev import FLUX_2_DEV, Flux2DevModel
from image_api_workers.generation_models import GenerationAdapterSettings, GenerationModels
from image_api_workers.generation_worker import create_worker_app
from test_ephemeral_safety_boundaries import _flux_2_klein_snapshot


def request(**changes):
    return {
        "model": FLUX_2_DEV,
        "prompt": "exact prompt",
        "seed": 42,
        "width": 256,
        "height": 272,
    } | changes


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = snapshot(tmp_path / "dev")
    heavy = HeavyBoundaries(monkeypatch, root)
    return root, heavy


def test_pinned_topology_and_default_configuration(prepared, monkeypatch):
    root, _ = prepared
    assert FLUX_2_DEV_MANIFEST["repository"] == "diffusers/FLUX.2-dev-bnb-4bit"
    assert FLUX_2_DEV_REVISION == "c30ad107542e63f222f864a8de510204394fb18a"
    assert flux_2_dev_weights_available(root)
    files = FLUX_2_DEV_MANIFEST["files"]
    assert (
        len([f for f in files if f.startswith("text_encoder/") and f.endswith(".safetensors")]) == 4
    )
    assert (
        len([f for f in files if f.startswith("transformer/") and f.endswith(".safetensors")]) == 2
    )
    assert "vae/diffusion_pytorch_model.safetensors" in files
    assert "vae/diffusion_pytorch_model.safetensors.index.json" not in files
    assert not (root / "tokenizer/merges.txt").exists()
    assert json.loads((root / "model_index.json").read_text())["tokenizer"] == [
        "transformers",
        "PixtralProcessor",
    ]
    monkeypatch.delenv("IMAGE_API_FLUX_2_DEV_WEIGHTS_PATH", raising=False)
    assert str(Settings.from_env().flux_2_dev_weights_path) == FLUX_2_DEV_MANIFEST["container_path"]
    monkeypatch.setenv("IMAGE_API_FLUX_2_DEV_WEIGHTS_PATH", str(root))
    assert Settings.from_env().flux_2_dev_weights_path == root


@pytest.mark.parametrize(
    "damage", ["revision", "quantization", "index", "shard", "direct-substitute", "tokenizer"]
)
def test_readiness_rejects_wrong_quantization_corruption_and_unsupported_topology(prepared, damage):
    root, heavy = prepared
    if damage == "revision":
        (root / ".image-api-revision").write_text("0" * 40)
    elif damage == "quantization":
        path = root / "text_encoder/config.json"
        config = json.loads(path.read_text())
        config["quantization_config"]["load_in_4bit"] = False
        path.write_text(json.dumps(config))
    elif damage == "index":
        (root / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text("{")
    elif damage == "shard":
        (root / "text_encoder/model-00004-of-00004.safetensors").write_bytes(b"truncated")
    elif damage == "direct-substitute":
        (root / "transformer/diffusion_pytorch_model.safetensors").write_bytes(b"unsupported")
    else:
        (root / "tokenizer/processor_config.json").unlink()
    assert not flux_2_dev_weights_available(root)
    with pytest.raises(RuntimeError, match="mount is incomplete"):
        Flux2DevModel(root)(request())
    assert not heavy.events


def test_actual_adapter_releases_both_stages_before_next_request(prepared):
    root, heavy = prepared
    adapter = Flux2DevModel(root)
    for _ in range(2):
        output = adapter(request())
        with Image.open(BytesIO(output)) as image:
            assert image.mode == "RGB" and image.size == (256, 272)
        assert not heavy.live
    meaningful = [event for event in heavy.events if event != "empty-cache"]
    assert (
        meaningful
        == [
            "load-encoder",
            "encode",
            "release-encoder",
            "load-transformer",
            "load-vae",
            "vae-cuda",
            "denoise",
            "release-transformer",
            "release-vae",
        ]
        * 2
    )
    adapter.unload()
    assert not heavy.live


@pytest.mark.parametrize("failure", ["load-encoder", "encode", "load-vae", "denoise"])
def test_actual_adapter_cleans_failed_stages_before_retry(prepared, failure, caplog):
    root, heavy = prepared
    adapter = Flux2DevModel(root)
    heavy.fail_at = failure
    with pytest.raises(RuntimeError, match="FLUX.2 dev generation failed"):
        adapter(request())
    assert not heavy.live
    assert not heavy.inference_active
    assert any(record.exc_info for record in caplog.records)
    heavy.fail_at = None
    assert adapter(request()).startswith(b"\x89PNG")
    assert not heavy.live


def test_no_cuda_fails_without_loading_or_cpu_fallback(prepared):
    root, heavy = prepared
    heavy.cuda_available = False
    with pytest.raises(RuntimeError, match="requires CUDA"):
        Flux2DevModel(root)(request())
    assert not heavy.events


def test_public_ingress_real_worker_child_and_adapter_are_connected(
    prepared, tmp_path, monkeypatch
):
    root, heavy = prepared
    settings = Settings.for_tests(tmp_path, flux_2_dev_weights_path=root)
    _flux_2_klein_snapshot(settings.flux_2_klein_4b_weights_path)
    processes = inline_processes(monkeypatch)
    lifecycle = []
    models = GenerationModels(
        GenerationAdapterSettings(
            "",
            (),
            str(settings.flux_2_klein_4b_weights_path),
            "",
            (),
            str(root),
        ),
        lifecycle_observer=lambda *event: lifecycle.append(event),
    )
    worker = TestClient(create_worker_app(models, settings))
    peer_calls = []

    def peer_transport(incoming):
        assert incoming.url.host in {"background-worker", "upscale-worker"}
        assert incoming.url.path == "/internal/unload"
        peer_calls.append(incoming.url.host)
        return httpx.Response(200, json={"unloaded": True})

    evictor = PeerEvictor(
        ("http://background-worker", "http://upscale-worker"),
        client_factory=lambda timeout: httpx.Client(transport=httpx.MockTransport(peer_transport)),
    )
    monkeypatch.setattr("image_api_workers.generation_worker._evict_peers", evictor)
    dispatched = []

    def transport(incoming):
        assert incoming.url.host in {"generation", "background", "upscale"}
        if incoming.url.host != "generation":
            assert incoming.url.path == "/health"
            return httpx.Response(200, json={"ready": True, "loaded": False})
        dispatched.append((incoming.method, incoming.url.path))
        response = worker.request(
            incoming.method,
            incoming.url.raw_path.decode(),
            content=incoming.read(),
            headers=dict(incoming.headers),
        )
        return httpx.Response(
            response.status_code, content=response.content, headers=response.headers
        )

    workers = HttpWorkerClient(
        "http://upscale",
        "http://background",
        900,
        1_000_000,
        httpx.MockTransport(transport),
        "http://generation",
    )
    coordinator = SingleFlightCoordinator()
    gateway = TestClient(create_app(settings=settings, workers=workers, coordinator=coordinator))
    catalog = gateway.get("/v1/models").json()["models"]
    assert {item["capability"] for item in catalog if item["model"] == FLUX_2_DEV} == {
        "generation",
        "image-editing",
    }
    health = gateway.get("/health").json()["capabilities"]["generation"]["models"]
    assert health[FLUX_2_DEV]["ready"] is True
    assert health["flux-2-klein-4b"]["ready"] is True

    # The real global lane remains owned throughout the actual adapter execution.
    def during_encoding():
        assert coordinator.status()["active"] == 1

    heavy.on_encode = during_encoding
    for _ in range(2):
        response = gateway.post("/v1/generations", json=request())
        assert response.status_code == 200, response.text
        assert Image.open(BytesIO(response.content)).size == (256, 272)
    assert len(processes) == 1
    assert len(peer_calls) == 2
    assert not heavy.live

    response = gateway.post(
        "/v1/image-edits",
        data={"model": FLUX_2_DEV, "prompt": "exact prompt", "seed": "42"},
        files={"file": ("source.png", png("RGBA", (768, 531)), "image/png")},
    )
    assert response.status_code == 200, response.text
    assert heavy.calls[-1] == (768, 531, True)
    assert Image.open(BytesIO(response.content)).size == (768, 528)
    assert coordinator.status()["active"] == 0

    # Distinct selection still follows the unchanged child exit/reap boundary.
    response = gateway.post("/v1/generations", json=request(model="flux-2-klein-4b"))
    assert response.status_code == 200, response.text
    assert processes[0].joined
    assert lifecycle[-4:] == [
        ("exit", FLUX_2_DEV, 0),
        ("reap", FLUX_2_DEV, 0),
        ("spawn", "flux-2-klein-4b", 1),
        ("load", "flux-2-klein-4b", 1),
    ]
    assert "existing-klein-offload" in heavy.events

    # Failure crosses the real child error result and reaps before next load.
    heavy.fail_at = "denoise"
    assert gateway.post("/v1/generations", json=request()).status_code == 502
    assert not models.child_alive and not heavy.live
    assert coordinator.status()["active"] == 0
    heavy.fail_at = None
    assert gateway.post("/v1/generations", json=request()).status_code == 200
    models.unload()
    assert all(process.joined for process in processes)
    assert not heavy.live
    assert ("POST", "/internal/generate") in dispatched
    assert ("POST", "/internal/image-edit") in dispatched

    # Optional model absence rejects only that selection before dispatch/loading.
    (root / ".image-api-revision").unlink()
    dispatch_count = len([item for item in dispatched if item[0] == "POST"])
    assert gateway.post("/v1/generations", json=request()).status_code == 503
    assert len([item for item in dispatched if item[0] == "POST"]) == dispatch_count
    assert gateway.post("/v1/generations", json=request(model="flux-2-klein-4b")).status_code == 200
    models.unload()

    # Invalid new-model requests cannot reach the worker.
    for changes in ({"prompt": None}, {"sampler_preset": "V4_DEFAULT_20"}, {"magic_prompt": True}):
        assert gateway.post("/v1/generations", json=request(**changes)).status_code == 422
    assert (
        gateway.post(
            "/v1/image-edits",
            data={
                "model": FLUX_2_DEV,
                "prompt": "exact prompt",
                "seed": "42",
                "negative_prompt": "no",
            },
            files={"file": ("source.png", png(), "image/png")},
        ).status_code
        == 422
    )
