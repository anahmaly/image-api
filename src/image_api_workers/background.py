from __future__ import annotations

import asyncio
import atexit
import gc
import io
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Literal, Protocol

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFilter

from image_api.images import validate_dimensions
from image_api.config import Settings
from image_api.lane import GpuLane
from image_api.processing import (
    ProcessingRunner,
    recover_processing_tasks,
    start_processing_runner,
)
from image_api.store import TaskStore
from image_api.workers import PeerEvictor
from image_api_workers.execution import execute_in_gpu_lane
from image_api_workers.uploads import read_bounded_upload

logger = logging.getLogger(__name__)
_active_model: str | None = None
_model_lock = threading.RLock()


class Sam2Adapter(Protocol):
    """Small injectable boundary around the locally installed SAM 2 runtime."""

    def predict_probability(
        self, source: Image.Image, box: tuple[int, int, int, int]
    ) -> Image.Image: ...
    def release(self) -> None: ...


class LocalSam2Adapter:
    def __init__(self, checkpoint: Path) -> None:
        if not checkpoint.is_file():
            raise RuntimeError("SAM guidance checkpoint is unavailable")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("SAM guidance runtime is unavailable") from exc
        self._predictor = SAM2ImagePredictor(
            build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(checkpoint), device="cuda")
        )

    def predict_probability(
        self, source: Image.Image, box: tuple[int, int, int, int]
    ) -> Image.Image:
        import numpy as np

        self._predictor.set_image(np.asarray(source.convert("RGB")))
        masks, scores, _ = self._predictor.predict(
            box=np.asarray(box), multimask_output=True, return_logits=True
        )
        score_index = max(range(len(scores)), key=lambda index: (float(scores[index]), -index))
        selected = np.asarray(masks[score_index], dtype=np.float32)
        probability = 1.0 / (1.0 + np.exp(-selected))
        return Image.fromarray(probability, mode="F")

    def release(self) -> None:
        self._predictor = None


_sam2_adapter: Sam2Adapter | None = None


def _create_sam2_adapter() -> Sam2Adapter:
    checkpoint = Path(
        os.getenv(
            "IMAGE_API_SAM2_CHECKPOINT_PATH",
            "/models/sam2.1-hiera-large/sam2.1_hiera_large.pt",
        )
    )
    return LocalSam2Adapter(checkpoint)


def sam2_prompt_box(alpha: Image.Image, threshold: int) -> tuple[int, int, int, int]:
    selected = alpha.convert("L").point(lambda value: 255 if value >= threshold else 0)
    bounds = selected.getbbox()
    if bounds is None:
        raise ValueError("SAM guidance prompt has no foreground pixels")
    left, top, right, bottom = bounds
    return left, top, right - 1, bottom - 1


def validate_sam2_mask(mask: Image.Image, expected_size: tuple[int, int]) -> None:
    if mask.size != expected_size:
        raise ValueError("SAM guidance mask dimensions are invalid")
    values = tuple(mask.getdata())
    if not values or any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= float(value) <= 1
        for value in values
    ):
        raise ValueError("SAM guidance mask contains invalid probabilities")


def _binary_sam2_mask(
    probability: Image.Image, threshold: float, size: tuple[int, int]
) -> Image.Image:
    validate_sam2_mask(probability, size)
    binary = bytes(255 if float(value) >= threshold else 0 for value in probability.getdata())
    if not any(binary) or all(binary):
        raise ValueError("SAM guidance mask must contain foreground and background")
    return Image.frombytes("L", size, binary)


def _fill_enclosed_holes(support: Image.Image) -> Image.Image:
    """Return binary support with only border-disconnected background filled."""
    background = support.point(lambda value: 255 if value == 0 else 0)
    width, height = support.size
    border_pixels = [
        *((x, 0) for x in range(width)),
        *((x, height - 1) for x in range(width)),
        *((0, y) for y in range(1, height - 1)),
        *((width - 1, y) for y in range(1, height - 1)),
    ]
    for coordinate in border_pixels:
        if background.getpixel(coordinate) == 255:
            ImageDraw.floodfill(background, coordinate, 128, border=0)
    return Image.frombytes(
        "L", support.size, bytes(0 if value == 128 else 255 for value in background.getdata())
    )


def _validate_monotonic_alpha(final_alpha: Image.Image, adjusted_alpha: Image.Image) -> None:
    if final_alpha.size != adjusted_alpha.size or any(
        final < adjusted for final, adjusted in zip(final_alpha.getdata(), adjusted_alpha.getdata())
    ):
        raise RuntimeError("SAM guidance produced destructive alpha")


def fuse_sam2_guidance(
    source: Image.Image,
    provisional_alpha: Image.Image,
    sam_mask: Image.Image,
    *,
    interior_erode: int,
    boundary_dilate: int,
    boundary_alpha_gamma: float,
    provisional_foreground_threshold: int,
) -> Image.Image:
    """Strengthen credible foreground without ever reducing adjusted matting alpha.

    ``boundary_dilate`` is a bounded binary closing radius. SAM support and
    high-confidence provisional support are unioned before closing; enclosed
    background is then filled before ``interior_erode`` defines the conservative
    core. Only that core becomes opaque, so adjusted provisional alpha stays soft
    wherever it is outside the conservative support.
    """
    size = source.size
    if provisional_alpha.size != size or sam_mask.size != size:
        raise ValueError("SAM guidance dimensions are invalid")
    if not 0 <= provisional_foreground_threshold <= 255:
        raise ValueError("SAM guidance provisional threshold is invalid")
    if interior_erode < 0 or boundary_dilate < 0:
        raise ValueError("SAM guidance morphology radius is invalid")

    lut = [
        min(255, max(0, int(255 * (value / 255) ** boundary_alpha_gamma + 0.5)))
        for value in range(256)
    ]
    adjusted_alpha = provisional_alpha.convert("L").point(lut)
    provisional_support = provisional_alpha.convert("L").point(
        lambda value: 255 if value >= provisional_foreground_threshold else 0
    )
    support = Image.frombytes(
        "L",
        size,
        bytes(
            255 if sam_value or provisional_value else 0
            for sam_value, provisional_value in zip(
                sam_mask.convert("L").getdata(), provisional_support.getdata()
            )
        ),
    )
    if boundary_dilate:
        kernel = boundary_dilate * 2 + 1
        support = support.filter(ImageFilter.MaxFilter(kernel)).filter(
            ImageFilter.MinFilter(kernel)
        )
    sure_foreground = _fill_enclosed_holes(support)
    if interior_erode:
        sure_foreground = sure_foreground.filter(ImageFilter.MinFilter(interior_erode * 2 + 1))

    alpha = adjusted_alpha.copy()
    alpha.paste(255, mask=sure_foreground)
    _validate_monotonic_alpha(alpha, adjusted_alpha)

    result = source.convert("RGBA")
    result.putalpha(alpha)
    hidden = Image.new("RGBA", size, (0, 0, 0, 0))
    result.paste(hidden, mask=alpha.point(lambda value: 255 if value == 0 else 0))
    return result


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


def _release_resident_models() -> None:
    global _active_model, _sam2_adapter
    with _model_lock:
        release_errors: list[BaseException] = []
        if _sam2_adapter is not None:
            try:
                _sam2_adapter.release()
            except Exception as exc:
                release_errors.append(exc)
                logger.exception("SAM guidance release failed")
            _sam2_adapter = None
        try:
            from rembg_api.birefnet_hr import clear_cache

            clear_cache()
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as exc:
            release_errors.append(exc)
            logger.exception("BiRefNet cache release failed")
        try:
            from rembg_api.bria_rmbg import clear_bria_backend_cache

            clear_bria_backend_cache(release_cuda_cache=True)
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as exc:
            release_errors.append(exc)
            logger.exception("BRIA cache release failed")
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as exc:
            release_errors.append(exc)
            logger.exception("background CUDA cache release failed")
        _active_model = None
        if release_errors:
            raise RuntimeError("background model release failed") from release_errors[0]


def _validate_worker_dimensions(width: int, height: int, *, output: bool) -> None:
    suffix = "OUTPUT" if output else "INPUT"
    validate_dimensions(
        width,
        height,
        max_width=int(os.getenv("IMAGE_API_PROCESSING_MAX_INPUT_WIDTH", "8192")),
        max_height=int(os.getenv("IMAGE_API_PROCESSING_MAX_INPUT_HEIGHT", "8192")),
        max_pixels=int(os.getenv(f"IMAGE_API_PROCESSING_MAX_{suffix}_PIXELS", "67108864")),
        max_decoded_bytes=int(
            os.getenv(f"IMAGE_API_PROCESSING_MAX_DECODED_{suffix}_BYTES", "268435456")
        ),
    )


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
    despill_enabled: bool = False,
    despill_color: str = "black",
    despill_hex_color: str = "000000",
    sam2_guidance: bool = False,
    sam2_model: str = "sam2.1-hiera-large",
    sam2_mask_threshold: float = 0.5,
    sam2_prompt_alpha_threshold: int = 128,
    sam2_interior_erode: int = 4,
    sam2_boundary_dilate: int = 8,
    boundary_alpha_gamma: float = 0.6,
) -> bytes:
    global _active_model
    if not 512 <= birefnet_inference_size <= 4096:
        raise ValueError("invalid BiRefNet inference size")
    with Image.open(io.BytesIO(data)) as source_image:
        expected_size = source_image.size
        _validate_worker_dimensions(*expected_size, output=False)
        _validate_worker_dimensions(*expected_size, output=True)
        source_image.verify()
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
        raise ValueError("unsupported background-removal model")
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
        despill=DespillOptions(
            enabled=despill_enabled,
            color=despill_color,
            hex_color=despill_hex_color,
        ),
        background_color="transparent",
        background_hex="ffffff",
        return_alpha=False,
        return_checker_preview=False,
        checker_size=32,
        max_encoded_bytes=int(
            os.getenv("IMAGE_API_PROCESSING_MAX_ENCODED_OUTPUT_BYTES", "300000000")
        ),
    )
    if not isinstance(encoded, bytes):
        raise RuntimeError("background post-processing returned invalid bytes")
    if sam2_guidance:
        if model != "birefnet-hr-matting" or sam2_model != "sam2.1-hiera-large":
            raise ValueError("SAM guidance requires the supported BiRefNet model")
        if not 0 <= sam2_mask_threshold <= 1 or not math.isfinite(sam2_mask_threshold):
            raise ValueError("invalid SAM guidance mask threshold")
        global _sam2_adapter
        with (
            Image.open(io.BytesIO(data)) as original,
            Image.open(io.BytesIO(encoded)) as processed,
        ):
            original.load()
            processed.load()
            adapter = _sam2_adapter or _create_sam2_adapter()
            _sam2_adapter = adapter
            probability = adapter.predict_probability(
                original.convert("RGB"),
                sam2_prompt_box(processed.getchannel("A"), sam2_prompt_alpha_threshold),
            )
            guided = fuse_sam2_guidance(
                original.convert("RGB"),
                processed.getchannel("A"),
                _binary_sam2_mask(probability, sam2_mask_threshold, expected_size),
                interior_erode=sam2_interior_erode,
                boundary_dilate=sam2_boundary_dilate,
                boundary_alpha_gamma=boundary_alpha_gamma,
                provisional_foreground_threshold=sam2_prompt_alpha_threshold,
            )
            guided_output = io.BytesIO()
            guided.save(guided_output, "PNG")
            encoded = guided_output.getvalue()
    with Image.open(io.BytesIO(encoded)) as output:
        output.load()
        if output.mode != "RGBA":
            raise RuntimeError("background backend did not return RGBA")
        if output.size != expected_size:
            raise RuntimeError("background backend returned unexpected dimensions")
    _active_model = model
    return encoded


def _health() -> dict[str, object]:
    bria = Path(os.getenv("IMAGE_API_BRIA_WEIGHTS_PATH", "/models/bria-rmbg-2.0"))
    birefnet = Path(os.getenv("IMAGE_API_BIREFNET_WEIGHTS_PATH", "/models/birefnet-hr"))
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception as exc:
        logger.warning(
            "background CUDA runtime probe failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        cuda = False
    loaded = _active_model is not None
    mounts = bria.is_dir() and birefnet.is_dir()
    return {
        "ready": cuda and mounts,
        "loaded": bool(loaded),
        "loadedModel": _active_model,
        "device": "cuda" if cuda else "unavailable",
        "weightsAvailable": mounts,
    }


def _shutdown_unload() -> None:
    try:
        _release_resident_models()
    except Exception:
        logger.exception("background worker shutdown unload failed")


atexit.register(_shutdown_unload)


def start_durable_runner() -> bool:
    if os.getenv("IMAGE_API_ENABLE_PROCESSING_RUNNER", "false").lower() != "true":
        return False

    def build_runner() -> ProcessingRunner:
        settings = Settings.from_env()
        store = TaskStore(
            settings.database_path,
            settings.max_queue_depth,
            processing_max_persisted_output_bytes=settings.processing_max_persisted_output_bytes,
            processing_max_encoded_output_bytes=settings.processing_max_encoded_output_bytes,
            output_dir=settings.output_dir,
        )
        recovered = recover_processing_tasks(
            "background-removal", store, settings.output_dir, settings.source_dir, settings
        )
        if recovered:
            logger.warning("reconciled interrupted background tasks: count=%s", recovered)

        def model(source: Path, request: dict[str, object]) -> bytes:
            with source.open("rb") as handle:
                data = handle.read(settings.processing_max_upload_bytes + 1)
            if len(data) > settings.processing_max_upload_bytes:
                raise ValueError("persisted source exceeds configured limit")
            model_name = request.get("model")
            alpha_blur = request.get("alpha_blur")
            alpha_erode = request.get("alpha_erode")
            alpha_dilate = request.get("alpha_dilate")
            alpha_threshold = request.get("alpha_threshold")
            inference_size = request.get("birefnet_inference_size")
            refinement = request.get("birefnet_foreground_refinement")
            model_input_size = request.get("model_input_size")
            despill_enabled = request.get("despill_enabled")
            despill_color = request.get("despill_color")
            despill_hex_color = request.get("despill_hex_color")
            sam2_guidance = request.get("sam2_guidance", False)
            sam2_model = request.get("sam2_model", "sam2.1-hiera-large")
            sam2_mask_threshold = request.get("sam2_mask_threshold", 0.5)
            sam2_prompt_alpha_threshold = request.get("sam2_prompt_alpha_threshold", 128)
            sam2_interior_erode = request.get("sam2_interior_erode", 4)
            sam2_boundary_dilate = request.get("sam2_boundary_dilate", 8)
            boundary_alpha_gamma = request.get("boundary_alpha_gamma", 0.6)
            if (
                model_name not in {"bria-rmbg-2.0", "birefnet-hr-matting"}
                or not isinstance(model_name, str)
                or not isinstance(alpha_blur, (int, float))
                or isinstance(alpha_blur, bool)
                or any(
                    type(value) is not int
                    for value in (
                        alpha_erode,
                        alpha_dilate,
                        alpha_threshold,
                        inference_size,
                        model_input_size,
                    )
                )
                or type(refinement) is not bool
                or type(despill_enabled) is not bool
                or despill_color not in {"black", "white", "green", "blue", "custom"}
                or not isinstance(despill_color, str)
                or not isinstance(despill_hex_color, str)
                or type(sam2_guidance) is not bool
                or sam2_model != "sam2.1-hiera-large"
                or not isinstance(sam2_model, str)
                or not isinstance(sam2_mask_threshold, (int, float))
                or isinstance(sam2_mask_threshold, bool)
                or not 0 <= float(sam2_mask_threshold) <= 1
                or not math.isfinite(float(sam2_mask_threshold))
                or type(sam2_prompt_alpha_threshold) is not int
                or not 1 <= sam2_prompt_alpha_threshold <= 255
                or type(sam2_interior_erode) is not int
                or not 0 <= sam2_interior_erode <= 64
                or type(sam2_boundary_dilate) is not int
                or not 0 <= sam2_boundary_dilate <= 64
                or not isinstance(boundary_alpha_gamma, (int, float))
                or isinstance(boundary_alpha_gamma, bool)
                or not 0.1 <= float(boundary_alpha_gamma) <= 4
                or not math.isfinite(float(boundary_alpha_gamma))
            ):
                raise ValueError("invalid persisted background-removal parameters")
            assert type(alpha_erode) is int
            assert type(alpha_dilate) is int
            assert type(alpha_threshold) is int
            assert type(inference_size) is int
            assert type(model_input_size) is int
            assert type(refinement) is bool
            assert type(despill_enabled) is bool
            assert type(sam2_guidance) is bool
            assert isinstance(sam2_model, str)
            assert type(sam2_prompt_alpha_threshold) is int
            assert type(sam2_interior_erode) is int
            assert type(sam2_boundary_dilate) is int
            with _model_lock:
                return _run_background(
                    data,
                    model=model_name,
                    alpha_blur=float(alpha_blur),
                    alpha_erode=alpha_erode,
                    alpha_dilate=alpha_dilate,
                    alpha_threshold=alpha_threshold,
                    birefnet_inference_size=inference_size,
                    birefnet_foreground_refinement=refinement,
                    model_input_size=model_input_size,
                    despill_enabled=despill_enabled,
                    despill_color=despill_color,
                    despill_hex_color=despill_hex_color,
                    sam2_guidance=sam2_guidance,
                    sam2_model=sam2_model,
                    sam2_mask_threshold=float(sam2_mask_threshold),
                    sam2_prompt_alpha_threshold=sam2_prompt_alpha_threshold,
                    sam2_interior_erode=sam2_interior_erode,
                    sam2_boundary_dilate=sam2_boundary_dilate,
                    boundary_alpha_gamma=float(boundary_alpha_gamma),
                )

        return ProcessingRunner(
            "background-removal",
            store,
            GpuLane(settings.gpu_lane_path, settings.lane_timeout_seconds),
            settings.source_dir,
            settings.output_dir,
            model,
            settings,
            peer_evictor=PeerEvictor(
                (
                    os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
                    os.getenv("IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"),
                )
            ),
        )

    poll = float(os.getenv("IMAGE_API_PROCESSING_POLL_SECONDS", "0.5"))
    backoff = float(os.getenv("IMAGE_API_PROCESSING_ERROR_BACKOFF_SECONDS", "1.0"))
    return start_processing_runner(
        "background-removal",
        build_runner,
        poll_seconds=poll,
        error_backoff_seconds=backoff,
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    start_durable_runner()
    yield


app = FastAPI(title="image-api-background-worker", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return _health()


@app.post("/internal/unload")
def unload() -> dict[str, object]:
    try:
        _release_resident_models()
    except Exception as exc:
        logger.exception("background worker unload failed")
        raise HTTPException(500, "model unload failed") from exc
    return {"unloaded": True, "loaded": False}


@app.post("/internal/background-removal", response_class=Response)
async def remove_background(
    file: Annotated[UploadFile, File()],
    model: Annotated[
        Literal["bria-rmbg-2.0", "birefnet-hr-matting"],
        Query(),
    ],
    alpha_blur: Annotated[float, Query(ge=0, le=20)] = 0,
    alpha_erode: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_dilate: Annotated[int, Query(ge=0, le=100)] = 0,
    alpha_threshold: Annotated[int, Query(ge=0, le=255)] = 0,
    birefnet_inference_size: Annotated[int, Query(ge=512, le=4096)] = 2048,
    birefnet_foreground_refinement: bool = False,
    model_input_size: Annotated[int, Query(ge=512, le=2048)] = 1024,
    sam2_guidance: bool = False,
    sam2_model: Literal["sam2.1-hiera-large"] = "sam2.1-hiera-large",
    sam2_mask_threshold: Annotated[float, Query(ge=0, le=1)] = 0.5,
    sam2_prompt_alpha_threshold: Annotated[int, Query(ge=1, le=255)] = 128,
    sam2_interior_erode: Annotated[int, Query(ge=0, le=64)] = 4,
    sam2_boundary_dilate: Annotated[int, Query(ge=0, le=64)] = 8,
    boundary_alpha_gamma: Annotated[float, Query(ge=0.1, le=4)] = 0.6,
) -> Response:
    max_upload_bytes = int(os.getenv("IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES", "280000000"))
    data = await read_bounded_upload(file, max_upload_bytes)
    try:

        def operation() -> bytes:
            PeerEvictor(
                (
                    os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
                    os.getenv("IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"),
                )
            )()
            with _model_lock:
                return _run_background(
                    data,
                    model=model,
                    alpha_blur=alpha_blur,
                    alpha_erode=alpha_erode,
                    alpha_dilate=alpha_dilate,
                    alpha_threshold=alpha_threshold,
                    birefnet_inference_size=birefnet_inference_size,
                    birefnet_foreground_refinement=birefnet_foreground_refinement,
                    model_input_size=model_input_size,
                    sam2_guidance=sam2_guidance,
                    sam2_model=sam2_model,
                    sam2_mask_threshold=sam2_mask_threshold,
                    sam2_prompt_alpha_threshold=sam2_prompt_alpha_threshold,
                    sam2_interior_erode=sam2_interior_erode,
                    sam2_boundary_dilate=sam2_boundary_dilate,
                    boundary_alpha_gamma=boundary_alpha_gamma,
                )

        encoded = await asyncio.to_thread(
            execute_in_gpu_lane,
            "background-removal",
            operation,
        )
        return Response(encoded, media_type="image/png")
    except Exception as exc:
        logger.exception("background worker failed: model=%s", model)
        raise HTTPException(500, "internal image processing error") from exc
