from __future__ import annotations

import json
import logging
import os
import re
from typing import Annotated, Any, Literal, cast

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from image_api.config import Settings, ideogram_weights_available
from image_api.generation import worker_heartbeat_alive
from image_api.images import (
    ImageTooLarge,
    InvalidImage,
    InvalidWorkerImage,
    validate_image,
    validate_png_output,
)
from image_api.lane import GpuLane, LaneBusy
from image_api.store import IdempotencyConflict, QueueFull, TaskRecord, TaskStore
from image_api.workers import HttpWorkerClient, WorkerClient, WorkerUnavailable

logger = logging.getLogger(__name__)

UPSCALE_MODELS = ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B")
BACKGROUND_MODELS = (
    "isnet-general-use",
    "u2net",
    "u2netp",
    "isnet-anime",
    "silueta",
    "bria-rmbg-2.0",
    "birefnet-hr-matting",
)
SAMPLER_PRESETS = ("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12")
TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
UPLOAD_CHUNK_BYTES = 64 * 1024


class RequestBodyTooLarge(BaseException):
    pass


class RequestBodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            if not declared.isdigit():
                await JSONResponse(
                    {"error": {"code": "invalid_request", "message": "Invalid request"}},
                    status_code=400,
                )(scope, receive, send)
                return
            if int(declared) > self.max_bytes:
                await JSONResponse(
                    {"error": {"code": "request_too_large", "message": "Request is too large"}},
                    status_code=413,
                )(scope, receive, send)
                return
        consumed = 0
        started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal consumed
            message = cast(dict[str, Any], await receive())
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if started:
                raise RuntimeError("request limit exceeded after response start")
            await JSONResponse(
                {"error": {"code": "request_too_large", "message": "Request is too large"}},
                status_code=413,
            )(scope, receive, send)


class GenerationRequest(BaseModel):
    width: int = Field(ge=256, le=2048, multiple_of=16)
    height: int = Field(ge=256, le=2048, multiple_of=16)
    seed: int = Field(ge=0, le=2**32 - 1)
    sampler_preset: Literal["V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"]
    structured_caption: dict[str, Any] | None = None
    prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    magic_prompt: bool = False

    @model_validator(mode="after")
    def validate_caption_mode(self) -> "GenerationRequest":
        structured = self.structured_caption is not None
        plain = self.prompt is not None
        if structured == plain:
            raise ValueError("provide exactly one caption mode")
        if structured:
            if not self.structured_caption:
                raise ValueError("structured_caption must be a non-empty JSON object")
            encoded = json.dumps(self.structured_caption, sort_keys=True, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > 64_000:
                raise ValueError("structured_caption is too large")
            if self.magic_prompt:
                raise ValueError("magic_prompt is only valid with prompt")
        elif not self.magic_prompt:
            raise ValueError("plain prompts require magic_prompt=true")
        return self


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > max_bytes:
                raise ImageTooLarge("upload exceeds configured limit")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        await file.close()


def _safe_task(task: TaskRecord) -> dict[str, object]:
    result: dict[str, object] = {
        "taskId": task.task_id,
        "status": task.status,
        "width": task.request["width"],
        "height": task.request["height"],
        "seed": task.request["seed"],
        "samplerPreset": task.request["sampler_preset"],
    }
    if task.error_code:
        result["error"] = {
            "code": task.error_code,
            "message": "Generation did not complete"
            if task.status == "failed"
            else "Generation unavailable",
        }
    return result


def _generation_health(settings: Settings) -> dict[str, object]:
    repository_id = os.getenv("IMAGE_API_IDEOGRAM_REPOSITORY_ID", "ideogram-ai/ideogram-4-nf4")
    mounted = ideogram_weights_available(settings.ideogram_weights_path, repository_id)
    worker_available = settings.generation_test_mode or worker_heartbeat_alive(
        settings.generation_heartbeat_path,
        max_age_seconds=settings.generation_heartbeat_max_age_seconds,
    )
    if settings.generation_test_mode:
        reason = None
    elif not mounted:
        reason = "weights_unavailable"
    elif not worker_available:
        reason = "worker_unavailable"
    elif not settings.cuda_available:
        reason = "cuda_unavailable"
    else:
        reason = None
    loaded = False
    model_state = "unloaded"
    try:
        status_path = settings.state_dir / "generation-model-status.json"
        with status_path.open() as handle:
            status = json.loads(handle.read(4096))
        if isinstance(status, dict) and status.get("state") in {"unloaded", "loading", "loaded"}:
            model_state = status["state"]
            loaded = bool(status.get("loaded", False))
    except (OSError, ValueError, TypeError):
        pass
    return {
        "ready": reason is None,
        "loaded": loaded,
        "modelState": model_state,
        "device": (
            "cpu-test"
            if settings.generation_test_mode
            else "cuda"
            if settings.cuda_available
            else "unavailable"
        ),
        "quantization": "nf4",
        "weightsAvailable": mounted,
        "workerAvailable": worker_available,
        "reason": reason,
    }


def create_app(
    *,
    settings: Settings | None = None,
    store: TaskStore | None = None,
    workers: WorkerClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    store = store or TaskStore(settings.database_path, settings.max_queue_depth)
    workers = workers or HttpWorkerClient(
        os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
        os.getenv("IMAGE_API_BACKGROUND_WORKER_URL", "http://background-worker:9002"),
        settings.worker_timeout_seconds,
        settings.max_request_bytes,
    )
    lane = GpuLane(settings.gpu_lane_path, settings.lane_timeout_seconds)

    app = FastAPI(title="image-api", version="1.0.0")
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.max_request_bytes)

    @app.exception_handler(WorkerUnavailable)
    async def worker_unavailable(_: Request, exc: WorkerUnavailable) -> JSONResponse:
        logger.error(
            "capability worker unavailable",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "worker_unavailable",
                    "message": "Image capability is temporarily unavailable",
                }
            },
        )

    @app.exception_handler(LaneBusy)
    async def lane_busy(_: Request, exc: LaneBusy) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {"code": "gpu_lane_busy", "message": "Image processing capacity is busy"}
            },
        )

    @app.exception_handler(ImageTooLarge)
    async def image_too_large(_: Request, exc: ImageTooLarge) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "image_too_large", "message": "Image exceeds accepted limits"}},
            status_code=413,
        )

    @app.exception_handler(InvalidImage)
    async def invalid_image(_: Request, exc: InvalidImage) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "invalid_image", "message": "Uploaded file is not a valid image"}},
            status_code=400,
        )

    @app.exception_handler(InvalidWorkerImage)
    async def invalid_worker_image(_: Request, exc: InvalidWorkerImage) -> JSONResponse:
        logger.error(
            "capability worker returned invalid output",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_worker_output",
                    "message": "Image capability returned invalid output",
                }
            },
            status_code=502,
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        raw_status = workers.health()
        capability_status: dict[str, dict[str, object]] = {}
        for capability in ("upscale", "background-removal"):
            worker_status = raw_status.get(capability, {})
            device = worker_status.get("device")
            if device not in {"cuda", "cpu-test", "unavailable"}:
                device = "unavailable"
            capability_status[capability] = {
                "ready": bool(worker_status.get("ready")),
                "loaded": bool(worker_status.get("loaded")),
                "device": device,
            }
            if "weightsAvailable" in worker_status:
                capability_status[capability]["weightsAvailable"] = bool(
                    worker_status["weightsAvailable"]
                )
            allowed_models = UPSCALE_MODELS if capability == "upscale" else BACKGROUND_MODELS
            loaded_model = worker_status.get("loadedModel")
            if loaded_model in allowed_models:
                capability_status[capability]["loadedModel"] = loaded_model
        capability_status["generation"] = _generation_health(settings)
        return {
            "service": "image-api",
            "status": "ok"
            if all(bool(v.get("ready")) for v in capability_status.values())
            else "degraded",
            "capabilities": capability_status,
            "gpuLane": lane.status(),
        }

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "models": [
                *({"capability": "upscale", "model": model} for model in UPSCALE_MODELS),
                *(
                    {"capability": "background-removal", "model": model}
                    for model in BACKGROUND_MODELS
                ),
                {
                    "capability": "generation",
                    "model": "ideogram-4-nf4",
                    "samplerPresets": list(SAMPLER_PRESETS),
                    "dimensions": {"minimum": 256, "maximum": 2048, "multipleOf": 16},
                },
            ]
        }

    @app.post("/v1/upscale", response_class=Response)
    async def upscale(
        file: Annotated[UploadFile, File()],
        model: Annotated[Literal["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], Query()],
        outscale: Annotated[float, Query(ge=1, le=4)],
        tile: Annotated[int, Query(ge=0, le=1024)],
    ) -> Response:
        if tile != 0 and tile % 32:
            raise HTTPException(422, "tile must be zero or a multiple of 32")
        data = await _read_upload(file, settings.max_upload_bytes)
        info = validate_image(
            data,
            max_bytes=settings.max_upload_bytes,
            max_width=settings.max_input_width,
            max_height=settings.max_input_height,
            max_pixels=settings.max_input_pixels,
        )
        expected = (round(info.width * outscale), round(info.height * outscale))
        if expected[0] * expected[1] > settings.max_output_pixels:
            raise ImageTooLarge("upscaled output exceeds configured limits")
        encoded = workers.upscale(data, model=model, outscale=outscale, tile=tile)
        validate_png_output(
            encoded,
            expected_size=expected,
            required_mode=None,
            max_bytes=settings.max_request_bytes,
            max_pixels=settings.max_output_pixels,
        )
        return Response(encoded, media_type="image/png")

    @app.post("/v1/background-removal", response_class=Response)
    async def background_removal(
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
        data = await _read_upload(file, settings.max_upload_bytes)
        info = validate_image(
            data,
            max_bytes=settings.max_upload_bytes,
            max_width=settings.max_input_width,
            max_height=settings.max_input_height,
            max_pixels=settings.max_input_pixels,
        )
        parameters = {
            "model": model,
            "alpha_blur": alpha_blur,
            "alpha_erode": alpha_erode,
            "alpha_dilate": alpha_dilate,
            "alpha_threshold": alpha_threshold,
            "birefnet_inference_size": birefnet_inference_size,
            "birefnet_foreground_refinement": birefnet_foreground_refinement,
            "model_input_size": model_input_size,
        }
        encoded = workers.background(data, **parameters)
        validate_png_output(
            encoded,
            expected_size=(info.width, info.height),
            required_mode="RGBA",
            max_bytes=settings.max_request_bytes,
            max_pixels=settings.max_output_pixels,
        )
        return Response(encoded, media_type="image/png")

    @app.post("/v1/generations", status_code=202)
    def admit_generation(
        body: GenerationRequest,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    ) -> dict[str, object]:
        if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
            raise HTTPException(422, "invalid idempotency key")
        if body.prompt is not None and settings.magic_prompt_backend is None:
            raise HTTPException(422, "plain prompt expansion is not configured")
        if not bool(_generation_health(settings)["ready"]):
            raise WorkerUnavailable("generation capability is unavailable")
        try:
            task = store.admit(idempotency_key, body.model_dump(exclude_none=True))
        except IdempotencyConflict as exc:
            raise HTTPException(409, "idempotency key conflicts with another request") from exc
        except QueueFull as exc:
            raise HTTPException(503, "generation queue is full") from exc
        return _safe_task(task)

    @app.get("/v1/generations/{task_id}")
    def generation_status(task_id: str) -> dict[str, object]:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(404, "task not found")
        try:
            return _safe_task(store.get(task_id))
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc

    @app.get("/v1/generations/{task_id}/image")
    def generation_image(task_id: str) -> FileResponse:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise HTTPException(404, "task not found")
        try:
            task = store.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, "task not found") from exc
        if task.status != "succeeded" or task.image_name != f"{task_id}.png":
            raise HTTPException(409, "generation image is not available")
        path = settings.output_dir / task.image_name
        if not path.is_file():
            logger.error("generation image missing for succeeded task")
            raise HTTPException(503, "generation image is unavailable")
        return FileResponse(path, media_type="image/png", filename=f"{task_id}.png")

    return app
