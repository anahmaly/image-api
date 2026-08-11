from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import uuid
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from image_api.config import (
    LONGCAT_EDIT_REVISION,
    LONGCAT_EDIT_TURBO_REVISION,
    longcat_weights_available,
)

logger = logging.getLogger(__name__)

LONGCAT_MODELS = ("longcat-image-edit", "longcat-image-edit-turbo")
OFFICIAL_DEFAULTS = {
    "longcat-image-edit": {"guidance_scale": 4.5, "num_inference_steps": 50},
    "longcat-image-edit-turbo": {"guidance_scale": 1.0, "num_inference_steps": 8},
}
OFFICIAL_REVISIONS = {
    "longcat-image-edit": LONGCAT_EDIT_REVISION,
    "longcat-image-edit-turbo": LONGCAT_EDIT_TURBO_REVISION,
}


class GenerationAdapter(Protocol):
    def __call__(self, request: dict[str, object]) -> bytes: ...
    def unload(self) -> None: ...


class LongCatRuntimeUnavailable(RuntimeError):
    pass


class LongCatImageEditModel:
    """Offline adapter for the official single-image LongCat edit pipeline."""

    def __init__(
        self,
        weights: dict[str, Path],
        source_dir: Path,
        *,
        revisions: dict[str, str] | None = None,
        pipeline_factory: Callable[[str, Path], Any] | None = None,
        generator_factory: Callable[[int], object] | None = None,
        cuda_available: Callable[[], bool] | None = None,
    ) -> None:
        self.weights = weights
        self.source_dir = source_dir
        self.revisions = revisions or OFFICIAL_REVISIONS
        self._pipeline_factory = pipeline_factory
        self._generator_factory = generator_factory
        self._cuda_available = cuda_available
        self._pipeline: Any | None = None
        self._loaded_model: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    def _load(self, model: str) -> Any:
        if self._loaded_model == model and self._pipeline is not None:
            return self._pipeline
        if self._pipeline is not None:
            self.unload()
        path = self.weights.get(model)
        if path is None or not path.is_dir():
            raise LongCatRuntimeUnavailable("configured LongCat weight mount is unavailable")
        if self._pipeline_factory is None and not longcat_weights_available(
            path, self.revisions[model]
        ):
            raise LongCatRuntimeUnavailable("configured LongCat weight mount is incomplete")
        cuda = self._cuda_available
        if cuda is None:
            import torch

            cuda = torch.cuda.is_available
        if not cuda():
            raise LongCatRuntimeUnavailable("LongCat image editing requires CUDA")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            if self._pipeline_factory is not None:
                pipeline = self._pipeline_factory(model, path)
            else:
                import torch
                from diffusers import LongCatImageEditPipeline

                pipeline = LongCatImageEditPipeline.from_pretrained(
                    str(path),
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                )
            pipeline.enable_model_cpu_offload()
        except Exception as exc:
            self._pipeline = None
            self._loaded_model = None
            logger.error(
                "LongCat runtime initialization failed: model=%s",
                model,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise LongCatRuntimeUnavailable("LongCat runtime initialization failed") from None
        self._pipeline = pipeline
        self._loaded_model = model
        return pipeline

    def _generator(self, seed: int) -> object:
        if self._generator_factory is not None:
            return self._generator_factory(seed)
        if self._pipeline_factory is not None:
            return ("cpu", seed)
        import torch

        return torch.Generator("cpu").manual_seed(seed)

    def __call__(self, request: dict[str, object]) -> bytes:
        with self._lock:
            model = request.get("model")
            source_name = request.get("source_image_name")
            source_bytes = request.get("source_image_bytes")
            prompt = request.get("prompt")
            negative_prompt = request.get("negative_prompt", "")
            seed = request.get("seed")
            if model not in LONGCAT_MODELS:
                raise ValueError("invalid persisted image-edit model")
            if not isinstance(source_bytes, bytes) and (
                not isinstance(source_name, str)
                or Path(source_name).name != source_name
                or not source_name.endswith(".png")
            ):
                raise ValueError("invalid persisted source image name")
            if not isinstance(prompt, str) or not 1 <= len(prompt) <= 4000:
                raise ValueError("invalid persisted image-edit prompt")
            if not isinstance(negative_prompt, str) or len(negative_prompt) > 4000:
                raise ValueError("invalid persisted negative prompt")
            if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                raise ValueError("invalid persisted image-edit seed")
            try:
                source_input = (
                    BytesIO(source_bytes)
                    if isinstance(source_bytes, bytes)
                    else self.source_dir / str(source_name)
                )
                with Image.open(source_input) as opened:
                    opened.load()
                    source_image = opened.convert("RGB")
            except (OSError, ValueError):
                raise LongCatRuntimeUnavailable("persisted source image is unavailable") from None
            pipeline = self._load(model)
            defaults = OFFICIAL_DEFAULTS[model]
            try:
                result = pipeline(
                    source_image,
                    prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=defaults["guidance_scale"],
                    num_inference_steps=defaults["num_inference_steps"],
                    num_images_per_prompt=1,
                    generator=self._generator(seed),
                )
                output_image = result.images[0].convert("RGB")
                if output_image.size != source_image.size:
                    output_image = output_image.resize(source_image.size)
            except Exception as exc:
                logger.error(
                    "LongCat inference failed: model=%s",
                    model,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise LongCatRuntimeUnavailable("LongCat inference failed") from None
            output = BytesIO()
            output_image.save(output, "PNG")
            return output.getvalue()

    def unload(self) -> None:
        with self._lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._loaded_model = None
            if pipeline is not None:
                for hook_name in ("_remove_all_hooks", "remove_all_hooks"):
                    remove_hooks = getattr(pipeline, hook_name, None)
                    if callable(remove_hooks):
                        remove_hooks()
                        break


def _generation_child(
    connection: object, ideogram: GenerationAdapter, longcat: LongCatImageEditModel
) -> None:
    """The child is the only process allowed to materialize a generation pipeline."""
    from multiprocessing.connection import Connection

    channel = connection
    assert isinstance(channel, Connection)
    while True:
        message = channel.recv()
        if message is None:
            return
        request = message
        target = request["model"]
        adapter: GenerationAdapter = ideogram if target == "ideogram-4-nf4" else longcat
        try:
            channel.send(("ok", adapter(request)))
        except Exception:  # noqa: BLE001 - child must serialize adapter failures for the parent
            channel.send(("error", "generation adapter failed"))


class GenerationModels:
    """Own exactly one model child; process exit is the cross-model memory boundary."""

    def __init__(
        self,
        ideogram: GenerationAdapter,
        longcat: LongCatImageEditModel,
        *,
        status_path: Path | None = None,
        lifecycle_observer: Callable[[str, str, int], None] | None = None,
    ) -> None:
        self.ideogram = ideogram
        self.longcat = longcat
        self.status_path = status_path
        self.loaded_model: str | None = None
        self._lock = threading.RLock()
        self._child: Any | None = None
        self._channel: object | None = None
        self._lifecycle_observer = lifecycle_observer
        self._write_status("unloaded")

    def _observe(self, event: str, model: str, live_children: int) -> None:
        if self._lifecycle_observer is not None:
            self._lifecycle_observer(event, model, live_children)

    def _write_status(self, state: str) -> None:
        if self.status_path is None:
            return
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, object] = {"state": state, "loaded": state == "loaded"}
        if state == "loaded" and self.loaded_model is not None:
            body["loadedModel"] = self.loaded_model
        temporary = self.status_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(body, sort_keys=True))
        os.replace(temporary, self.status_path)

    @property
    def child_alive(self) -> bool:
        return self._child is not None and bool(self._child.is_alive())

    def __call__(self, request: dict[str, object]) -> bytes:
        target = request.get("model", "ideogram-4-nf4")
        if target != "ideogram-4-nf4" and target not in LONGCAT_MODELS:
            raise ValueError("invalid persisted generation model")
        with self._lock:
            if self.loaded_model is not None and self.loaded_model != target:
                self.unload()
            self._write_status("loading")
            if self._child is None:
                parent, child = multiprocessing.Pipe()
                self._channel = parent
                self._child = multiprocessing.get_context("fork").Process(
                    target=_generation_child, args=(child, self.ideogram, self.longcat), daemon=True
                )
                self._child.start()
                child.close()
                self._observe("spawn", str(target), 1)
            try:
                from multiprocessing.connection import Connection

                channel = self._channel
                assert isinstance(channel, Connection)
                channel.send(request | {"model": target})
                kind, result = channel.recv()
                if kind != "ok" or not isinstance(result, bytes):
                    raise RuntimeError(str(result))
                encoded = result
            except Exception:
                self.unload()
                raise
            self.loaded_model = str(target)
            self._observe("load", self.loaded_model, 1)
            self._write_status("loaded")
            return encoded

    def unload(self) -> None:
        with self._lock:
            child = self._child
            channel = self._channel
            if child is not None:
                model = self.loaded_model
                try:
                    from multiprocessing.connection import Connection

                    if isinstance(channel, Connection):
                        channel.send(None)
                        channel.close()
                except (BrokenPipeError, EOFError, OSError):
                    pass
                child.join()
                if child.is_alive():
                    child.terminate()
                    child.join()
                if model is not None:
                    self._observe("exit", model, 0)
                    self._observe("reap", model, 0)
            self._child = None
            self._channel = None
            self.loaded_model = None
            self._write_status("unloaded")
