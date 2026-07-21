from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from image_api_workers.execution import execute_in_gpu_lane

logger = logging.getLogger(__name__)
REMBG_MODELS = ("isnet-general-use", "u2net", "u2netp", "isnet-anime", "silueta")
REMBG_FILES = {model: f"{model}.onnx" for model in REMBG_MODELS}
_active_model: str | None = None


def _birefnet_config() -> Any:
    from rembg_api.birefnet_hr import BiRefNetConfig, DEFAULT_REVISION

    return BiRefNetConfig(
        source=os.getenv("IMAGE_API_BIREFNET_WEIGHTS_PATH", "/models/birefnet-hr"),
        revision=os.getenv("IMAGE_API_BIREFNET_REVISION", DEFAULT_REVISION),
        local_files_only=True,
        trust_remote_code=True,
        cache_dir=None,
        device="cuda",
        precision="fp16",
        inference_size=2048,
        foreground_refinement=False,
        max_concurrency=1,
    )


@lru_cache(maxsize=len(REMBG_MODELS))
def _rembg_session(model: str) -> Any:
    from rembg import new_session

    weights = Path(os.getenv("IMAGE_API_REMBG_WEIGHTS_PATH", "/models/rembg"))
    expected = weights / REMBG_FILES[model]
    if not expected.is_file():
        raise FileNotFoundError("configured rembg model mount is unavailable")
    os.environ["U2NET_HOME"] = str(weights)
    return new_session(model)


def _release_resident_models() -> None:
    global _active_model
    _rembg_session.cache_clear()
    try:
        from rembg_api.birefnet_hr import clear_cache

        clear_cache()
    except Exception:
        pass
    try:
        from rembg_api.bria_rmbg import clear_bria_backend_cache

        clear_bria_backend_cache(release_cuda_cache=True)
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _active_model = None


def _run_background(
    data: bytes,
    *,
    model: str,
    alpha_blur: float,
    alpha_erode: int,
    alpha_dilate: int,
    alpha_threshold: int,
    birefnet_inference_size: int,
    birefnet_foreground_refinement: bool,
    model_input_size: int,
) -> bytes:
    global _active_model
    if _active_model is not None and _active_model != model:
        _release_resident_models()
    if model == "birefnet-hr-matting":
        from rembg_api.birefnet_hr import remove_with_birefnet

        removed = remove_with_birefnet(
            data,
            inference_size=birefnet_inference_size,
            foreground_refinement=birefnet_foreground_refinement,
            config=_birefnet_config(),
        )
    elif model == "bria-rmbg-2.0":
        from rembg_api.bria_rmbg import remove_with_bria_rmbg_2

        removed = remove_with_bria_rmbg_2(
            data,
            model_input_size=model_input_size,
            device="cuda",
            dtype="fp16",
            model_path=os.getenv("IMAGE_API_BRIA_WEIGHTS_PATH", "/models/bria-rmbg-2.0"),
        )
    else:
        from rembg import remove

        removed = remove(data, session=_rembg_session(model))
    if not isinstance(removed, bytes):
        raise RuntimeError("background backend returned invalid bytes")
    from rembg_api.image_processing import AlphaOptions, DespillOptions, process_png_bytes

    encoded = process_png_bytes(
        removed,
        alpha=AlphaOptions(
            blur=alpha_blur,
            erode=alpha_erode,
            dilate=alpha_dilate,
            threshold=alpha_threshold,
        ),
        despill=DespillOptions(),
        background_color="transparent",
        background_hex="ffffff",
        return_alpha=False,
        return_checker_preview=False,
        checker_size=32,
        max_encoded_bytes=40_000_000,
    )
    if not isinstance(encoded, bytes):
        raise RuntimeError("background post-processing returned invalid bytes")
    with Image.open(io.BytesIO(encoded)) as output:
        output.load()
        if output.mode != "RGBA":
            raise RuntimeError("background backend did not return RGBA")
    _active_model = model
    return encoded


def _health() -> dict[str, object]:
    bria = Path(os.getenv("IMAGE_API_BRIA_WEIGHTS_PATH", "/models/bria-rmbg-2.0"))
    birefnet = Path(os.getenv("IMAGE_API_BIREFNET_WEIGHTS_PATH", "/models/birefnet-hr"))
    rembg = Path(os.getenv("IMAGE_API_REMBG_WEIGHTS_PATH", "/models/rembg"))
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    loaded = _active_model is not None
    mounts = (
        bria.is_dir()
        and birefnet.is_dir()
        and all((rembg / filename).is_file() for filename in REMBG_FILES.values())
    )
    return {
        "ready": cuda and mounts,
        "loaded": bool(loaded),
        "loadedModel": _active_model,
        "device": "cuda" if cuda else "unavailable",
        "weightsAvailable": mounts,
    }


app = FastAPI(title="image-api-background-worker", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, object]:
    return _health()


@app.post("/internal/background-removal", response_class=Response)
async def remove_background(
    file: Annotated[UploadFile, File()],
    model: Annotated[
        Literal[
            "isnet-general-use",
            "u2net",
            "u2netp",
            "isnet-anime",
            "silueta",
            "bria-rmbg-2.0",
            "birefnet-hr-matting",
        ],
        Query(),
    ],
    alpha_blur: Annotated[float, Query(ge=0, le=20)] = 0,
    alpha_erode: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_dilate: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_threshold: Annotated[int, Query(ge=0, le=255)] = 0,
    birefnet_inference_size: Annotated[int, Query(ge=512, le=4096)] = 2048,
    birefnet_foreground_refinement: bool = False,
    model_input_size: Annotated[int, Query(ge=512, le=2048)] = 1024,
) -> Response:
    data = await file.read()
    await file.close()
    try:
        encoded = await asyncio.to_thread(
            execute_in_gpu_lane,
            "background-removal",
            lambda: _run_background(
                data,
                model=model,
                alpha_blur=alpha_blur,
                alpha_erode=alpha_erode,
                alpha_dilate=alpha_dilate,
                alpha_threshold=alpha_threshold,
                birefnet_inference_size=birefnet_inference_size,
                birefnet_foreground_refinement=birefnet_foreground_refinement,
                model_input_size=model_input_size,
            ),
        )
        return Response(encoded, media_type="image/png")
    except Exception as exc:
        logger.exception("background worker failed: model=%s", model)
        raise HTTPException(500, "internal image processing error") from exc
