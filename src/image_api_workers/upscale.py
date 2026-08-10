from __future__ import annotations

import asyncio
import atexit
import gc
import io
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from image_api.images import validate_dimensions
from image_api_workers.execution import execute_in_gpu_lane
from image_api_workers.uploads import read_bounded_upload

MODELS = {
    "RealESRGAN_x4plus": (23, "RealESRGAN_x4plus.pth"),
    "RealESRGAN_x4plus_anime_6B": (6, "RealESRGAN_x4plus_anime_6B.pth"),
}
_loaded_model_name: str | None = None
_model_lock = threading.RLock()


def _weights_dir() -> Path:
    return Path(os.getenv("IMAGE_API_UPSCALE_WEIGHTS_PATH", "/models/upscale"))


def _runtime_status() -> dict[str, object]:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return {
        "ready": cuda and all((_weights_dir() / name).is_file() for _, name in MODELS.values()),
        "loaded": _loaded_model_name is not None,
        "loadedModel": _loaded_model_name,
        "device": "cuda" if cuda else "unavailable",
        "weightsAvailable": all((_weights_dir() / name).is_file() for _, name in MODELS.values()),
    }


@lru_cache(maxsize=1)
def _load_model(model: str) -> Any:
    global _loaded_model_name
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    blocks, filename = MODELS[model]
    if not torch.cuda.is_available() or not (_weights_dir() / filename).is_file():
        raise RuntimeError("upscale runtime unavailable")
    _loaded_model_name = model
    return RealESRGANer(
        scale=4,
        model_path=str(_weights_dir() / filename),
        model=RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=blocks, num_grow_ch=32, scale=4
        ),
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=torch.device("cuda"),
    )


def _release() -> None:
    global _loaded_model_name
    with _model_lock:
        _load_model.cache_clear()
        _loaded_model_name = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


atexit.register(_release)


def _run(data: bytes, model: str, outscale: float, tile: int) -> bytes:
    import cv2
    import numpy as np
    import torch

    if model not in MODELS or not 1 <= outscale <= 4 or (tile and tile % 32):
        raise ValueError("invalid upscale parameters")
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        width, height = source.size
        validate_dimensions(
            width,
            height,
            max_width=int(os.getenv("IMAGE_API_PROCESSING_MAX_INPUT_WIDTH", "8192")),
            max_height=int(os.getenv("IMAGE_API_PROCESSING_MAX_INPUT_HEIGHT", "8192")),
            max_pixels=int(os.getenv("IMAGE_API_PROCESSING_MAX_INPUT_PIXELS", "67108864")),
            max_decoded_bytes=int(
                os.getenv("IMAGE_API_PROCESSING_MAX_DECODED_INPUT_BYTES", "268435456")
            ),
        )
        image = source.convert("RGB")
    backend = _load_model(model)
    old = backend.tile_size
    backend.tile_size = tile
    try:
        with torch.inference_mode():
            result, _ = backend.enhance(np.asarray(image)[:, :, ::-1], outscale=outscale)
    finally:
        backend.tile_size = old
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return bytes(encoded)


app = FastAPI(title="image-api-upscale-worker", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, object]:
    return _runtime_status()


@app.post("/internal/unload")
def unload() -> dict[str, object]:
    _release()
    return {"unloaded": True, "loaded": False}


@app.post("/internal/upscale")
async def upscale(
    file: Annotated[UploadFile, File()],
    model: Annotated[Literal["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], Query()],
    outscale: Annotated[float, Query(ge=1, le=4)],
    tile: Annotated[int, Query(ge=0, le=1024)],
) -> Response:
    data = await read_bounded_upload(
        file, int(os.getenv("IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES", "280000000"))
    )
    try:
        return Response(
            await asyncio.to_thread(
                execute_in_gpu_lane, "upscale", lambda: _run(data, model, outscale, tile)
            ),
            media_type="image/png",
        )
    except Exception as exc:
        raise HTTPException(500, "internal image processing error") from exc
