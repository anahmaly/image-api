from __future__ import annotations

import multiprocessing
from io import BytesIO
from pathlib import Path

from PIL import Image

from image_api_workers.generation_models import GenerationAdapterSettings


class _IdeogramAdapter:
    def __call__(self, request: dict[str, object]) -> bytes:
        caption = request.get("structured_caption")
        if caption == {"description": "post-entry"}:
            raise RuntimeError("fixture output failure after handler entry")
        width = request.get("width", 8)
        height = request.get("height", 6)
        assert type(width) is int and type(height) is int
        output = BytesIO()
        Image.new("RGB", (width, height)).save(output, "PNG")
        return output.getvalue()

    def unload(self) -> None:
        pass


class _LongCatAdapter:
    def __call__(self, request: dict[str, object]) -> bytes:
        if request.get("prompt") == "ordinary-failure":
            raise RuntimeError("fixture model failure after handler entry")
        source = request.get("source_image_bytes")
        assert isinstance(source, bytes)
        return source

    def unload(self) -> None:
        pass


class _Flux2KleinAdapter:
    def __call__(self, request: dict[str, object]) -> bytes:
        prompt = request.get("prompt")
        assert isinstance(prompt, str)
        source = request.get("source_image_bytes")
        if isinstance(source, bytes):
            with Image.open(BytesIO(source)) as image:
                output = BytesIO()
                image.convert("RGB").save(output, "PNG")
                return output.getvalue()
        width = request.get("width")
        height = request.get("height")
        assert type(width) is int and type(height) is int
        output = BytesIO()
        Image.new("RGB", (width, height)).save(output, "PNG")
        return output.getvalue()

    def unload(self) -> None:
        pass


def build_adapters(
    settings: GenerationAdapterSettings,
) -> tuple[_IdeogramAdapter, _LongCatAdapter, _Flux2KleinAdapter]:
    if settings.ideogram_weights_path:
        Path(settings.ideogram_weights_path).write_text(multiprocessing.get_start_method())
    return _IdeogramAdapter(), _LongCatAdapter(), _Flux2KleinAdapter()


def settings() -> GenerationAdapterSettings:
    return GenerationAdapterSettings("", (), "", "", ())
