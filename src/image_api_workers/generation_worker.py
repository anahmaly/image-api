from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from image_api.config import Settings, ideogram_weights_available, longcat_weights_available
from image_api_workers.generation_models import GenerationModels, LongCatImageEditModel
from image_api_workers.ideogram import IdeogramModel

logging.basicConfig(level=os.getenv("IMAGE_API_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def create_worker_app(models: GenerationModels, settings: Settings) -> FastAPI:
    app = FastAPI(title="image-api-generation-worker", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, object]:
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except (ImportError, AttributeError, RuntimeError):
            cuda = False
        repository = os.getenv("IMAGE_API_IDEOGRAM_REPOSITORY_ID", "ideogram-ai/ideogram-4-nf4")
        mounts = {
            "ideogram-4-nf4": ideogram_weights_available(
                settings.ideogram_weights_path, repository
            ),
            "longcat-image-edit": longcat_weights_available(
                settings.longcat_edit_weights_path, settings.longcat_edit_revision
            ),
            "longcat-image-edit-turbo": longcat_weights_available(
                settings.longcat_edit_turbo_weights_path, settings.longcat_edit_turbo_revision
            ),
        }
        return {
            "ready": cuda and all(mounts.values()),
            "loaded": models.loaded_model is not None,
            "device": "cuda" if cuda else "unavailable",
            "weightsAvailable": all(mounts.values()),
            "models": {
                name: {"weightsAvailable": value, "loaded": models.loaded_model == name}
                for name, value in mounts.items()
            },
            "loadedModel": models.loaded_model,
        }

    @app.post("/internal/unload")
    def unload() -> dict[str, object]:
        models.unload()
        return {"unloaded": True, "loaded": False}

    @app.post("/internal/generate")
    def generate(request: dict[str, object]) -> Response:
        try:
            return Response(models(request), media_type="image/png")
        except Exception as exc:
            logger.exception("generation worker failed")
            raise HTTPException(500, "internal generation error") from exc

    @app.post("/internal/image-edit")
    async def edit(
        file: UploadFile = File(),
        model: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 0,
    ) -> Response:
        try:
            data = await file.read()
            return Response(
                models(
                    {
                        "model": model,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": seed,
                        "source_image_bytes": data,
                    }
                ),
                media_type="image/png",
            )
        except Exception as exc:
            logger.exception("image edit worker failed")
            raise HTTPException(500, "internal generation error") from exc
        finally:
            await file.close()

    return app


def main() -> None:
    settings = Settings.from_env()
    models = GenerationModels(
        IdeogramModel(settings.ideogram_weights_path),
        LongCatImageEditModel(
            {
                "longcat-image-edit": settings.longcat_edit_weights_path,
                "longcat-image-edit-turbo": settings.longcat_edit_turbo_weights_path,
            },
            Path("/tmp"),
            revisions={
                "longcat-image-edit": settings.longcat_edit_revision,
                "longcat-image-edit-turbo": settings.longcat_edit_turbo_revision,
            },
        ),
    )
    atexit.register(models.unload)
    import uvicorn

    uvicorn.run(
        create_worker_app(models, settings),
        host="0.0.0.0",
        port=int(os.getenv("IMAGE_API_GENERATION_WORKER_PORT", "9003")),
        log_level=os.getenv("IMAGE_API_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
