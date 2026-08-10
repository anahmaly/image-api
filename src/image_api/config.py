from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPOSITORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SNAPSHOT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
LONGCAT_EDIT_REVISION = "7b54ef423aa7854be7861600024be5c56ab7875a"
LONGCAT_EDIT_TURBO_REVISION = "6a7262de5549f0bf0ec54c08ef7d283ef41f3214"
SQUARE_8K_EDGE = 8192


def _weights_available(directory: Path, filename: str) -> bool:
    return (directory / filename).is_file() or (directory / f"{filename}.index.json").is_file()


def ideogram_weights_available(weights_path: Path, repository_id: str) -> bool:
    if not REPOSITORY_ID_PATTERN.fullmatch(repository_id):
        return False
    reference = (
        weights_path / "hub" / f"models--{repository_id.replace('/', '--')}" / "refs" / "main"
    )
    try:
        snapshot = reference.read_text().strip()
    except OSError:
        return False
    root = reference.parent.parent / "snapshots" / snapshot
    return SNAPSHOT_PATTERN.fullmatch(snapshot) is not None and all(
        (root / item).is_file()
        for item in (
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "tokenizer/tokenizer.json",
        )
    )


def longcat_weights_available(weights_path: Path, revision: str) -> bool:
    try:
        installed = (weights_path / ".image-api-revision").read_text().strip()
    except OSError:
        return False
    return (
        installed == revision
        and (weights_path / "config.json").is_file()
        and _weights_available(weights_path / "transformer", "diffusion_pytorch_model.safetensors")
    )


@dataclass(frozen=True)
class Settings:
    ideogram_weights_path: Path
    longcat_edit_weights_path: Path
    longcat_edit_turbo_weights_path: Path
    longcat_edit_revision: str = LONGCAT_EDIT_REVISION
    longcat_edit_turbo_revision: str = LONGCAT_EDIT_TURBO_REVISION
    max_upload_bytes: int = 20_000_000
    max_input_width: int = 10_000
    max_input_height: int = 10_000
    max_input_pixels: int = 40_000_000
    max_decoded_input_bytes: int = 160_000_000
    processing_max_upload_bytes: int = 280_000_000
    processing_max_encoded_output_bytes: int = 300_000_000
    processing_max_input_width: int = SQUARE_8K_EDGE
    processing_max_input_height: int = SQUARE_8K_EDGE
    processing_max_input_pixels: int = SQUARE_8K_EDGE * SQUARE_8K_EDGE
    processing_max_output_pixels: int = SQUARE_8K_EDGE * SQUARE_8K_EDGE
    processing_max_decoded_input_bytes: int = SQUARE_8K_EDGE * SQUARE_8K_EDGE * 4
    processing_max_decoded_output_bytes: int = SQUARE_8K_EDGE * SQUARE_8K_EDGE * 4
    processing_max_native_width: int = 16384
    processing_max_native_height: int = 16384
    processing_max_native_pixels: int = 268435456
    processing_max_native_bytes: int = 3221225472
    worker_timeout_seconds: float = 900.0
    magic_prompt_backend: str | None = None
    generation_test_mode: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ideogram_weights_path=Path(
                os.getenv("IMAGE_API_IDEOGRAM_WEIGHTS_PATH", "/models/ideogram-4-nf4")
            ),
            longcat_edit_weights_path=Path(
                os.getenv("IMAGE_API_LONGCAT_EDIT_WEIGHTS_PATH", "/models/longcat-image-edit")
            ),
            longcat_edit_turbo_weights_path=Path(
                os.getenv(
                    "IMAGE_API_LONGCAT_EDIT_TURBO_WEIGHTS_PATH", "/models/longcat-image-edit-turbo"
                )
            ),
            worker_timeout_seconds=float(os.getenv("IMAGE_API_WORKER_TIMEOUT_SECONDS", "900")),
            magic_prompt_backend=os.getenv("IMAGE_API_MAGIC_PROMPT_BACKEND") or None,
            generation_test_mode=os.getenv("IMAGE_API_GENERATION_TEST_MODE", "false").lower()
            == "true",
        )

    @classmethod
    def for_tests(cls, root: Path, **overrides: Any) -> Settings:
        value = cls(
            ideogram_weights_path=root / "ideogram",
            longcat_edit_weights_path=root / "longcat",
            longcat_edit_turbo_weights_path=root / "longcat-turbo",
            max_upload_bytes=1_000_000,
            max_input_pixels=1_000_000,
            max_decoded_input_bytes=4_000_000,
            processing_max_upload_bytes=1_000_000,
            processing_max_encoded_output_bytes=1_000_000,
            processing_max_input_pixels=1_000_000,
            processing_max_output_pixels=4_000_000,
            processing_max_decoded_input_bytes=4_000_000,
            processing_max_decoded_output_bytes=16_000_000,
            generation_test_mode=True,
        )
        return replace(value, **overrides)

    def admit_upscale_processing(self, width: int, height: int) -> tuple[int, int]:
        if (
            width * 4 > self.processing_max_native_width
            or height * 4 > self.processing_max_native_height
        ):
            raise ValueError("Real-ESRGAN native processing dimensions exceed configured limits")
        return width * 4, height * 4
