from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any, BinaryIO, Literal, cast

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from image_api.config import Settings
from image_api.coordinator import CoordinatorBusy, SingleFlightCoordinator
from image_api.images import (
    ImageTooLarge,
    InvalidImage,
    InvalidWorkerImage,
    processing_output_size,
    validate_image,
    validate_png_output,
)
from image_api.workers import (
    HttpWorkerClient,
    WorkerClient,
    WorkerExecutionFailed,
    WorkerUnavailable,
)

logger = logging.getLogger(__name__)
UPSCALE_MODELS = ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B")
BACKGROUND_MODELS = ("bria-rmbg-2.0", "birefnet-hr-matting")
SAMPLER_PRESETS = ("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12")
LONGCAT_MODELS = ("longcat-image-edit", "longcat-image-edit-turbo")
FLUX_2_KLEIN_4B = "flux-2-klein-4b"


class GenerationRequest(BaseModel):
    model: Literal["ideogram-4-nf4", "flux-2-klein-4b"] = "ideogram-4-nf4"
    width: int = Field(ge=256, le=2048, multiple_of=16)
    height: int = Field(ge=256, le=2048, multiple_of=16)
    seed: int = Field(ge=0, le=2**32 - 1)
    sampler_preset: Literal["V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"] | None = None
    structured_caption: dict[str, Any] | None = None
    prompt: str | None = Field(default=None, min_length=1, max_length=4000)
    magic_prompt: bool = False

    @model_validator(mode="after")
    def validate_caption_mode(self) -> GenerationRequest:
        if self.model == FLUX_2_KLEIN_4B:
            if (
                self.prompt is None
                or self.structured_caption is not None
                or self.magic_prompt
                or self.sampler_preset is not None
            ):
                raise ValueError("FLUX.2 Klein requires a direct plain prompt")
            return self
        if self.sampler_preset is None:
            raise ValueError("Ideogram generation requires a sampler_preset")
        if (self.structured_caption is None) == (self.prompt is None):
            raise ValueError("provide exactly one caption mode")
        if self.structured_caption is not None:
            if not self.structured_caption or self.magic_prompt:
                raise ValueError(
                    "structured_caption must be a non-empty JSON object without magic_prompt"
                )
            if len(json.dumps(self.structured_caption, sort_keys=True).encode()) > 64_000:
                raise ValueError("structured_caption is too large")
        elif not self.magic_prompt:
            raise ValueError("plain prompts require magic_prompt=true")
        return self


async def _read_upload(file: UploadFile, maximum: int) -> bytes:
    try:
        data = await file.read(maximum + 1)
        if len(data) > maximum:
            raise ImageTooLarge("upload exceeds configured limit")
        return data
    finally:
        await file.close()


class RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject request bytes before Starlette parses or spools multipart parts."""

    def __init__(self, app: Any, default_max_bytes: int, route_max_bytes: dict[str, int]) -> None:
        self.app = app
        self.default_max_bytes = default_max_bytes
        self.route_max_bytes = route_max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        path = scope.get("path")
        maximum = (
            self.route_max_bytes.get(path, self.default_max_bytes)
            if isinstance(path, str)
            else self.default_max_bytes
        )
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            if not declared.isdigit():
                await JSONResponse(
                    {"error": {"code": "invalid_request", "message": "Invalid request"}},
                    status_code=400,
                )(scope, receive, send)
                return
            if int(declared) > maximum:
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
                if consumed > maximum:
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


def _bytes(output: object) -> bytes:
    if isinstance(output, bytes):
        return output
    if not all(hasattr(output, item) for item in ("read", "seek", "close")):
        raise InvalidWorkerImage("worker output type mismatch")
    stream = cast(BinaryIO, output)
    try:
        stream.seek(0)
        return stream.read()
    finally:
        stream.close()


def _generation_status(status: dict[str, object]) -> dict[str, object]:
    required_models = ("ideogram-4-nf4", FLUX_2_KLEIN_4B, *LONGCAT_MODELS)
    raw_models = status.get("models")
    model_weights = {
        model: type(raw_models.get(model, {}).get("weightsAvailable")) is bool
        and raw_models.get(model, {}).get("weightsAvailable") is True
        if isinstance(raw_models, dict) and isinstance(raw_models.get(model), dict)
        else False
        for model in required_models
    }
    weights = (
        type(status.get("weightsAvailable")) is bool
        and status.get("weightsAvailable") is True
        and all(model_weights.values())
    )
    ready = type(status.get("ready")) is bool and status.get("ready") is True and weights
    worker_available = (
        type(status.get("workerReachable")) is bool
        and status.get("workerReachable") is True
        and type(status.get("healthSchemaValid")) is bool
        and status.get("healthSchemaValid") is True
        and status.get("device") in {"cuda", "cpu-test"}
    )
    return {
        "ready": ready,
        "loaded": type(status.get("loaded")) is bool and status.get("loaded") is True,
        "device": status.get("device", "unavailable"),
        "weightsAvailable": weights,
        "workerAvailable": worker_available,
        "models": {
            model: {
                "weightsAvailable": available,
                "ready": worker_available and available,
            }
            for model, available in model_weights.items()
        },
    }


def create_app(
    *,
    settings: Settings | None = None,
    workers: WorkerClient | None = None,
    coordinator: SingleFlightCoordinator | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    workers = workers or HttpWorkerClient(
        os.getenv("IMAGE_API_UPSCALE_WORKER_URL", "http://upscale-worker:9001"),
        os.getenv("IMAGE_API_BACKGROUND_WORKER_URL", "http://background-worker:9002"),
        settings.worker_timeout_seconds,
        settings.processing_max_encoded_output_bytes,
        generation_url=os.getenv(
            "IMAGE_API_GENERATION_WORKER_URL", "http://generation-worker:9003"
        ),
    )
    coordinator = coordinator or SingleFlightCoordinator()
    app = FastAPI(title="image-api", version="1.0.0")
    app.add_middleware(
        RequestBodyLimitMiddleware,
        default_max_bytes=settings.max_request_bytes,
        route_max_bytes={
            "/v1/upscale": settings.processing_max_request_bytes,
            "/v1/background-removal": settings.processing_max_request_bytes,
        },
    )

    @app.exception_handler(WorkerUnavailable)
    async def unavailable(_: Request, exc: WorkerUnavailable) -> JSONResponse:
        logger.warning("internal worker unavailable", exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse(
            {
                "error": {
                    "code": "worker_unavailable",
                    "message": "Image capability is temporarily unavailable",
                }
            },
            status_code=503,
        )

    @app.exception_handler(CoordinatorBusy)
    async def busy(_: Request, exc: CoordinatorBusy) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": "image_capacity_busy",
                    "message": "Image processing capacity is busy",
                }
            },
            status_code=503,
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(WorkerExecutionFailed)
    async def execution_failed(_: Request, exc: WorkerExecutionFailed) -> JSONResponse:
        logger.warning(
            "internal worker execution result is ambiguous",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            {
                "error": {
                    "code": "worker_execution_unknown",
                    "message": "Image execution outcome is unknown",
                }
            },
            status_code=502,
        )

    @app.exception_handler(ImageTooLarge)
    async def too_large(_: Request, __: ImageTooLarge) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "image_too_large", "message": "Image exceeds accepted limits"}},
            status_code=413,
        )

    @app.exception_handler(InvalidImage)
    async def invalid(_: Request, __: InvalidImage) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "invalid_image", "message": "Uploaded file is not a valid image"}},
            status_code=400,
        )

    @app.exception_handler(InvalidWorkerImage)
    async def invalid_worker_output(_: Request, __: InvalidWorkerImage) -> JSONResponse:
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
        raw = workers.health()
        generation = _generation_status(raw.get("generation", {}))
        capabilities: dict[str, dict[str, object]] = {"generation": generation}
        for name in ("upscale", "background-removal"):
            status = raw.get(name, {})
            capabilities[name] = {
                "ready": bool(status.get("ready")),
                "loaded": bool(status.get("loaded")),
                "device": status.get("device", "unavailable"),
            }
        return {
            "service": "image-api",
            "status": "ok" if all(v["ready"] for v in capabilities.values()) else "degraded",
            "capabilities": capabilities,
            "coordinator": coordinator.status(),
        }

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "models": (
                [{"capability": "upscale", "model": model} for model in UPSCALE_MODELS]
                + [
                    {"capability": "background-removal", "model": model}
                    for model in BACKGROUND_MODELS
                ]
                + [
                    {
                        "capability": "generation",
                        "model": "ideogram-4-nf4",
                        "acceptsSourceImage": False,
                        "samplerPresets": list(SAMPLER_PRESETS),
                    }
                ]
                + [
                    {
                        "capability": "generation",
                        "model": FLUX_2_KLEIN_4B,
                        "acceptsSourceImage": False,
                    }
                ]
                + [
                    {
                        "capability": "image-editing",
                        "model": model,
                        "acceptsSourceImage": True,
                        "inputImages": 1,
                    }
                    for model in LONGCAT_MODELS
                ]
            )
        }

    @app.post("/v1/models/unload")
    def unload() -> JSONResponse:
        result = coordinator.run(workers.unload_all)
        complete = all(bool(item.get("unloaded")) for item in result.values())
        return JSONResponse(
            {"unloaded": complete, "workers": result}, status_code=200 if complete else 503
        )

    @app.post("/v1/upscale", response_class=Response)
    async def upscale(
        file: Annotated[UploadFile, File()],
        model: Annotated[Literal["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"], Query()],
        outscale: Annotated[float, Query(ge=1, le=4)],
        tile: Annotated[int, Query(ge=0, le=1024)],
    ) -> Response:
        if tile and tile % 32:
            raise HTTPException(422, "tile must be zero or a multiple of 32")
        data = await _read_upload(file, settings.processing_max_upload_bytes)
        info = validate_image(
            data,
            max_bytes=settings.processing_max_upload_bytes,
            max_width=settings.processing_max_input_width,
            max_height=settings.processing_max_input_height,
            max_pixels=settings.processing_max_input_pixels,
            max_decoded_bytes=settings.processing_max_decoded_input_bytes,
        )
        expected = processing_output_size(info, outscale)
        settings.admit_upscale_processing(info.width, info.height)
        encoded = _bytes(
            coordinator.run(
                lambda: workers.upscale(data, model=model, outscale=outscale, tile=tile)
            )
        )
        validate_png_output(
            encoded,
            expected_size=expected,
            required_mode="RGB",
            max_bytes=settings.processing_max_encoded_output_bytes,
            max_pixels=settings.processing_max_output_pixels,
            max_decoded_bytes=settings.processing_max_decoded_output_bytes,
        )
        return Response(encoded, media_type="image/png")

    @app.post("/v1/background-removal", response_class=Response)
    async def background(
        file: Annotated[UploadFile, File()],
        model: Annotated[Literal["bria-rmbg-2.0", "birefnet-hr-matting"], Query()],
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
        data = await _read_upload(file, settings.processing_max_upload_bytes)
        info = validate_image(
            data,
            max_bytes=settings.processing_max_upload_bytes,
            max_width=settings.processing_max_input_width,
            max_height=settings.processing_max_input_height,
            max_pixels=settings.processing_max_input_pixels,
            max_decoded_bytes=settings.processing_max_decoded_input_bytes,
        )
        params = {
            "model": model,
            "alpha_blur": alpha_blur,
            "alpha_erode": alpha_erode,
            "alpha_dilate": alpha_dilate,
            "alpha_threshold": alpha_threshold,
            "birefnet_inference_size": birefnet_inference_size,
            "birefnet_foreground_refinement": birefnet_foreground_refinement,
            "model_input_size": model_input_size,
            "sam2_guidance": sam2_guidance,
            "sam2_model": sam2_model,
            "sam2_mask_threshold": sam2_mask_threshold,
            "sam2_prompt_alpha_threshold": sam2_prompt_alpha_threshold,
            "sam2_interior_erode": sam2_interior_erode,
            "sam2_boundary_dilate": sam2_boundary_dilate,
            "boundary_alpha_gamma": boundary_alpha_gamma,
        }
        encoded = _bytes(coordinator.run(lambda: workers.background(data, **params)))
        validate_png_output(
            encoded,
            expected_size=(info.width, info.height),
            required_mode="RGBA",
            max_bytes=settings.processing_max_encoded_output_bytes,
            max_pixels=settings.processing_max_output_pixels,
            max_decoded_bytes=settings.processing_max_decoded_output_bytes,
        )
        return Response(encoded, media_type="image/png")

    @app.post("/v1/generations", response_class=Response)
    def generation(body: GenerationRequest) -> Response:
        if (
            body.model != FLUX_2_KLEIN_4B
            and body.prompt is not None
            and settings.magic_prompt_backend is None
        ):
            raise HTTPException(422, "plain prompt expansion is not configured")
        generation = _generation_status(workers.health().get("generation", {}))
        model_matrix = cast(dict[str, dict[str, bool]], generation["models"])
        if not model_matrix[body.model]["ready"]:
            raise WorkerUnavailable("generation capability is unavailable")
        encoded = _bytes(
            coordinator.run(lambda: workers.generation(body.model_dump(exclude_none=True)))
        )
        validate_png_output(
            encoded,
            expected_size=(body.width, body.height),
            required_mode="RGB",
            max_bytes=100_000_000,
            max_pixels=body.width * body.height,
        )
        return Response(encoded, media_type="image/png")

    @app.post("/v1/image-edits", response_class=Response)
    async def image_edit(
        file: Annotated[UploadFile, File()],
        model: Annotated[
            Literal["longcat-image-edit", "longcat-image-edit-turbo", "flux-2-klein-4b"], Form()
        ],
        prompt: Annotated[str, Form(min_length=1, max_length=4000)],
        seed: Annotated[int, Form(ge=0, le=2**32 - 1)],
        negative_prompt: Annotated[str, Form(max_length=4000)] = "",
    ) -> Response:
        generation = _generation_status(workers.health().get("generation", {}))
        model_matrix = cast(dict[str, dict[str, bool]], generation["models"])
        if not model_matrix[model]["ready"]:
            raise WorkerUnavailable("selected image-edit model is unavailable")
        data = await _read_upload(file, settings.max_upload_bytes)
        info = validate_image(
            data,
            max_bytes=settings.max_upload_bytes,
            max_width=settings.max_input_width,
            max_height=settings.max_input_height,
            max_pixels=settings.max_input_pixels,
            max_decoded_bytes=settings.max_decoded_input_bytes,
        )
        encoded = _bytes(
            coordinator.run(
                lambda: workers.image_edit(
                    data, model=model, prompt=prompt, negative_prompt=negative_prompt, seed=seed
                )
            )
        )
        validate_png_output(
            encoded,
            expected_size=(info.width, info.height),
            required_mode="RGB",
            max_bytes=100_000_000,
            max_pixels=info.width * info.height,
        )
        return Response(encoded, media_type="image/png")

    return app
