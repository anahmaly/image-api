from __future__ import annotations

import gc
import logging
import threading
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from PIL import Image

from image_api.config import flux_2_dev_weights_available
from image_api.generation_options import resolve_num_inference_steps, validate_vae_tiling

logger = logging.getLogger(__name__)
FLUX_2_DEV = "flux-2-dev-bnb-4bit"


def _release_cuda_memory() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


class Flux2DevModel:
    """Pinned NF4 encoder -> embeddings -> NF4 denoiser, with GPU-only inference.

    No pipeline survives a request. In particular, the next encoder must never be
    loaded beside the previous request's denoiser. The owning child/process and
    gateway lane remain the cross-model unload and exclusive-execution authority.
    """

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = weights_path
        self._lock = threading.RLock()

    def _encode(self, prompt: str) -> Any:
        import torch
        from diffusers import Flux2Pipeline
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        encoder = pipeline = None
        try:
            encoder = Mistral3ForConditionalGeneration.from_pretrained(
                str(self.weights_path / "text_encoder"),
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
            )
            processor = AutoProcessor.from_pretrained(
                str(self.weights_path / "tokenizer"), local_files_only=True
            )
            pipeline = Flux2Pipeline(
                scheduler=None,
                vae=None,
                text_encoder=encoder,
                tokenizer=processor,
                transformer=None,
            )
            # encode_prompt is not itself decorated with no_grad upstream. Without
            # this scope the retained embedding could retain the entire encoder.
            with torch.inference_mode():
                embeddings, _ = pipeline.encode_prompt(prompt=prompt, device="cuda:0")
            return embeddings
        finally:
            pipeline = encoder = None
            _release_cuda_memory()

    def _generate(
        self, embeddings: Any, parameters: dict[str, Any], vae_tiling: bool
    ) -> Image.Image:
        import torch
        from diffusers import AutoencoderKLFlux2, Flux2Pipeline, Flux2Transformer2DModel

        transformer = vae = pipeline = None
        try:
            transformer = Flux2Transformer2DModel.from_pretrained(
                str(self.weights_path / "transformer"),
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
            )
            vae = AutoencoderKLFlux2.from_pretrained(
                str(self.weights_path / "vae"),
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            ).to("cuda:0")
            # This fresh request-owned VAE handles both reference encoding and
            # output decoding. Transformer diffusion remains full-latent.
            if vae_tiling:
                vae.enable_tiling()
            pipeline = Flux2Pipeline.from_pretrained(
                str(self.weights_path),
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                text_encoder=None,
                tokenizer=None,
                transformer=transformer,
                vae=vae,
            )
            with torch.inference_mode():
                result = (
                    pipeline(
                        prompt_embeds=embeddings,
                        caption_upsample_temperature=0,
                        guidance_scale=4.0,
                        num_images_per_prompt=1,
                        **parameters,
                    )
                    .images[0]
                    .convert("RGB")
                )
                return cast(Image.Image, result)
        finally:
            pipeline = transformer = vae = None
            _release_cuda_memory()

    def __call__(self, request: dict[str, object]) -> bytes:
        with self._lock:
            prompt, seed = request.get("prompt"), request.get("seed")
            if request.get("model") != FLUX_2_DEV:
                raise ValueError("invalid FLUX.2 dev model")
            vae_tiling = validate_vae_tiling(FLUX_2_DEV, request.get("vae_tiling", False))
            steps = resolve_num_inference_steps(FLUX_2_DEV, request)
            if not isinstance(prompt, str) or not 1 <= len(prompt) <= 4000:
                raise ValueError("invalid FLUX.2 dev prompt")
            if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                raise ValueError("invalid FLUX.2 dev seed")
            if request.get("negative_prompt"):
                raise ValueError("FLUX.2 dev does not support negative prompts")
            parameters: dict[str, Any] = {"num_inference_steps": steps}
            source = request.get("source_image_bytes")
            if isinstance(source, bytes):
                with Image.open(BytesIO(source)) as opened:
                    opened.load()
                    parameters["image"] = opened.convert("RGB")
                    parameters["width"], parameters["height"] = opened.size
            else:
                width, height = request.get("width"), request.get("height")
                if any(
                    type(n) is not int or not 256 <= n <= 2048 or n % 16 for n in (width, height)
                ):
                    raise ValueError("invalid FLUX.2 dev dimensions")
                parameters.update(width=width, height=height)
            if not flux_2_dev_weights_available(self.weights_path):
                raise RuntimeError("configured FLUX.2 dev weight mount is incomplete")
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("FLUX.2 dev requires CUDA")
            embeddings = None
            image = None
            try:
                embeddings = self._encode(prompt)
                parameters["generator"] = torch.Generator(device="cuda:0").manual_seed(seed)
                image = self._generate(embeddings, parameters, vae_tiling)
            except Exception as exc:
                logger.error("FLUX.2 dev generation failed", exc_info=exc)
                traceback.clear_frames(exc.__traceback__)
            # Leave the exception scope before collecting: failed loader/inference
            # tracebacks may own GPU tensors. No failed stage can survive a retry.
            embeddings = None
            parameters.clear()
            _release_cuda_memory()
            if image is None:
                raise RuntimeError("FLUX.2 dev generation failed") from None
            output = BytesIO()
            image.save(output, "PNG")
            return output.getvalue()

    def unload(self) -> None:
        # Requests own and release all tensors, including on failure. Acquire the
        # same lock so explicit unload cannot race a still-running request.
        with self._lock:
            pass
