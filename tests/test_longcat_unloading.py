from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from helpers import png
from image_api_workers.generation_models import GenerationModels, LongCatImageEditModel


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
    events: list[object] = []
    longcat = LongCatImageEditModel(
        {"longcat-image-edit": tmp_path},
        tmp_path,
        pipeline_factory=lambda model, _path: EditPipeline(model, events),
        cuda_available=lambda: True,
    )
    models = GenerationModels(FakeIdeogram(events), longcat)
    models(
        {
            "model": "longcat-image-edit",
            "source_image_bytes": png(),
            "prompt": "x",
            "negative_prompt": "",
            "seed": 1,
        }
    )
    models({"model": "ideogram-4-nf4", "width": 256, "height": 256, "seed": 1})
    assert events[-2:] == [("hooks", "longcat-image-edit"), ("ideogram", 1)]
    assert models.loaded_model == "ideogram-4-nf4"
