from __future__ import annotations

from fastapi.testclient import TestClient

from helpers import png
from image_api.app import create_app
from image_api.config import Settings, ideogram_weights_available
from image_api.store import TaskStore
from image_api.workers import WorkerUnavailable


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


def test_ideogram_mount_requires_a_complete_offline_cache_shape(tmp_path) -> None:
    repository_id = "ideogram-ai/ideogram-4-nf4"
    assert ideogram_weights_available(tmp_path, repository_id) is False
    snapshot_name = "a" * 40
    repository = tmp_path / "hub" / "models--ideogram-ai--ideogram-4-nf4"
    reference = repository / "refs" / "main"
    reference.parent.mkdir(parents=True)
    reference.write_text(snapshot_name)
    snapshot = repository / "snapshots" / snapshot_name
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
    for directory in ("transformer", "unconditional_transformer"):
        index = snapshot / directory / "diffusion_pytorch_model.safetensors.index.json"
        shard = snapshot / directory / "weights-00001.safetensors"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text('{"weight_map":{"layer":"weights-00001.safetensors"}}')
        shard.write_bytes(b"weights")
    assert ideogram_weights_available(tmp_path, repository_id) is True
