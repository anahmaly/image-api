from __future__ import annotations

import sys
import types
from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient
from helpers import png
from PIL import Image
from spawn_adapters import build_adapters
from spawn_adapters import settings as spawn_settings

from image_api.config import Settings
from image_api_workers.generation_models import (
    Flux2KleinModel,
    GenerationAdapterSettings,
    GenerationModels,
    LongCatImageEditModel,
)
from image_api_workers.generation_worker import create_worker_app


class EditPipeline:
    def __init__(self, name: str, events: list[object]) -> None:
        self.name = name
        self.events = events
        self.hooks_removed = 0

    def enable_model_cpu_offload(self) -> None:
        self.events.append(("offload", self.name))

    def remove_all_hooks(self) -> None:
        self.hooks_removed += 1
        self.events.append(("hooks", self.name))

    def __call__(self, image: Image.Image, prompt: str, **kwargs: object):
        self.events.append(("call", self.name, image.copy(), prompt, kwargs))
        return SimpleNamespace(images=[Image.new("RGB", (13, 7))])


class FakeIdeogram:
    loaded = False

    def __init__(self, events: list[object]) -> None:
        self.events = events

    def __call__(self, request: dict[str, object]) -> bytes:
        self.loaded = True
        self.events.append(("ideogram", request["seed"]))
        return png("RGB", (256, 256))

    def unload(self) -> None:
        self.loaded = False
        self.events.append("unload-ideogram")


class FluxPipeline:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def enable_model_cpu_offload(self) -> None:
        self.events.append("offload")

    def remove_all_hooks(self) -> None:
        self.events.append("hooks")

    def __call__(self, **parameters: object):
        self.events.append(parameters)
        return SimpleNamespace(images=[Image.new("RGBA", (13, 7))])


def test_flux_2_klein_forwards_exact_prompt_and_source_as_rgb(tmp_path) -> None:
    events: list[object] = []
    adapter = Flux2KleinModel(
        tmp_path,
        pipeline_factory=lambda _path: FluxPipeline(events),
        generator_factory=lambda seed: ("cuda", seed),
        cuda_available=lambda: True,
    )
    output = adapter(
        {
            "model": "flux-2-klein-4b",
            "source_image_bytes": png("RGBA", (13, 7)),
            "prompt": "exact edit prompt",
            "seed": 43,
        }
    )

    call = next(event for event in events if isinstance(event, dict))
    assert call["prompt"] == "exact edit prompt"
    assert call["generator"] == ("cuda", 43)
    assert isinstance(call["image"], Image.Image)
    assert call["image"].mode == "RGB"
    with Image.open(BytesIO(output)) as image:
        assert image.mode == "RGB"
    adapter.unload()
    assert events[-1] == "hooks"


def test_longcat_uses_official_defaults_and_releases_on_model_switch(tmp_path) -> None:
    events: list[object] = []
    weights = {"longcat-image-edit": tmp_path, "longcat-image-edit-turbo": tmp_path}
    adapter = LongCatImageEditModel(
        weights,
        tmp_path,
        pipeline_factory=lambda model, _path: EditPipeline(model, events),
        cuda_available=lambda: True,
    )
    request: dict[str, object] = {
        "model": "longcat-image-edit",
        "source_image_bytes": png("RGB", (13, 7)),
        "prompt": "blue",
        "negative_prompt": "",
        "seed": 43,
    }
    adapter(request)
    adapter(request | {"model": "longcat-image-edit-turbo"})
    standard = next(
        event
        for event in events
        if isinstance(event, tuple) and event[:2] == ("call", "longcat-image-edit")
    )
    turbo = next(
        event
        for event in events
        if isinstance(event, tuple) and event[:2] == ("call", "longcat-image-edit-turbo")
    )
    assert standard[4] == {
        "negative_prompt": "",
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "num_images_per_prompt": 1,
        "generator": ("cpu", 43),
    }
    assert turbo[4] == standard[4] | {"guidance_scale": 1.0, "num_inference_steps": 8}
    assert ("hooks", "longcat-image-edit") in events


def test_generation_model_switch_unloads_longcat_before_ideogram(tmp_path) -> None:
    models = GenerationModels(spawn_settings(), adapter_factory=build_adapters)
    models(
        {
            "model": "longcat-image-edit",
            "source_image_bytes": png(),
            "prompt": "x",
            "negative_prompt": "",
            "seed": 1,
        }
    )
    assert models._child is not None
    first_child = models._child
    first_pid = first_child.pid
    models(
        {
            "model": "longcat-image-edit-turbo",
            "source_image_bytes": png(),
            "prompt": "x",
            "negative_prompt": "",
            "seed": 1,
        }
    )
    assert models._child is not None
    assert models._child.pid != first_pid
    assert first_child.is_alive() is False
    assert models.child_alive is True
    assert models.loaded_model == "longcat-image-edit-turbo"
    models.unload()


def test_generation_model_switch_reaps_flux_before_longcat() -> None:
    lifecycle: list[tuple[str, str, int]] = []
    models = GenerationModels(
        spawn_settings(),
        adapter_factory=build_adapters,
        lifecycle_observer=lambda event, model, live_children: lifecycle.append(
            (event, model, live_children)
        ),
    )
    models(
        {
            "model": "flux-2-klein-4b",
            "width": 256,
            "height": 256,
            "prompt": "exact generation prompt",
            "seed": 1,
        }
    )
    models(
        {
            "model": "longcat-image-edit",
            "source_image_bytes": png(),
            "prompt": "edit",
            "negative_prompt": "",
            "seed": 1,
        }
    )
    assert lifecycle[:6] == [
        ("spawn", "flux-2-klein-4b", 1),
        ("load", "flux-2-klein-4b", 1),
        ("exit", "flux-2-klein-4b", 0),
        ("reap", "flux-2-klein-4b", 0),
        ("spawn", "longcat-image-edit", 1),
        ("load", "longcat-image-edit", 1),
    ]
    assert max(live_children for _, _, live_children in lifecycle) == 1
    models.unload()


def test_generation_unload_refuses_success_while_a_child_remains_alive(tmp_path) -> None:
    class UnreapedChild:
        def __init__(self) -> None:
            self.events: list[str] = []

        def join(self, timeout: float) -> None:
            assert timeout == 30.0
            self.events.append("join")

        def terminate(self) -> None:
            self.events.append("terminate")

        def is_alive(self) -> bool:
            return True

    child = UnreapedChild()
    models = GenerationModels(spawn_settings(), adapter_factory=build_adapters)
    models._child = child
    models.loaded_model = "flux-2-klein-4b"

    response = TestClient(
        create_worker_app(models, Settings.for_tests(tmp_path)), raise_server_exceptions=False
    ).post("/internal/unload")

    assert response.status_code == 500
    assert child.events == ["join", "terminate", "join"]
    assert models.child_alive is True
    assert models.loaded_model == "flux-2-klein-4b"


def test_generation_switch_does_not_spawn_the_target_when_old_child_cannot_be_reaped() -> None:
    class UnreapedChild:
        def __init__(self) -> None:
            self.events: list[str] = []

        def join(self, timeout: float) -> None:
            assert timeout == 30.0
            self.events.append("join")

        def terminate(self) -> None:
            self.events.append("terminate")

        def is_alive(self) -> bool:
            return True

    child = UnreapedChild()
    models = GenerationModels(spawn_settings(), adapter_factory=build_adapters)
    models._child = child
    models.loaded_model = "flux-2-klein-4b"

    try:
        models(
            {
                "model": "longcat-image-edit",
                "source_image_bytes": png(),
                "prompt": "edit",
                "negative_prompt": "",
                "seed": 1,
            }
        )
    except RuntimeError as exc:
        assert str(exc) == "generation model child did not terminate"
    else:
        raise AssertionError("target model loaded before the old child was reaped")

    assert child.events == ["join", "terminate", "join"]
    assert models.child_alive is True
    assert models.loaded_model == "flux-2-klein-4b"


def test_generation_child_uses_spawn_after_parent_cuda_inspection(monkeypatch, tmp_path) -> None:
    """CUDA-sensitive construction happens in a fresh spawned interpreter, not a fork clone."""
    child_start_method = tmp_path / "child-start-method"
    cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    assert cuda.is_available() is True
    models = GenerationModels(
        GenerationAdapterSettings(str(child_start_method), (), "", "", ()),
        adapter_factory=build_adapters,
    )

    assert models(
        {
            "model": "ideogram-4-nf4",
            "width": 256,
            "height": 256,
            "seed": 1,
            "sampler_preset": "V4_TURBO_12",
            "structured_caption": {"description": "spawn-safe"},
        }
    ).startswith(b"\x89PNG")
    assert child_start_method.read_text() == "spawn"
    models.unload()
