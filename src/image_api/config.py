from __future__ import annotations

import json
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
MAX_SNAPSHOT_MARKER_BYTES = 65
MAX_SNAPSHOT_JSON_BYTES = 12_000_000
MAX_SNAPSHOT_TEXT_BYTES = 1_000_000


def _read_bounded(path: Path, maximum: int) -> bytes | None:
    try:
        if path.stat().st_size > maximum:
            return None
        with path.open("rb") as source:
            data = source.read(maximum + 1)
    except OSError:
        return None
    return data if len(data) <= maximum else None


def _json_object_available(path: Path) -> bool:
    data = _read_bounded(path, MAX_SNAPSHOT_JSON_BYTES)
    if not data:
        return False
    try:
        return isinstance(json.loads(data), dict)
    except (UnicodeDecodeError, ValueError):
        return False


def _text_available(path: Path) -> bool:
    data = _read_bounded(path, MAX_SNAPSHOT_TEXT_BYTES)
    if not data:
        return False
    try:
        return bool(data.decode().strip())
    except UnicodeDecodeError:
        return False


def _weight_index_available(index_path: Path) -> bool:
    """Accept only bounded, local shard indexes with every referenced shard present."""
    data = _read_bounded(index_path, MAX_SNAPSHOT_JSON_BYTES)
    if not data:
        return False
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, ValueError):
        return False
    weight_map = document.get("weight_map") if isinstance(document, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    for shard in set(weight_map.values()):
        if not isinstance(shard, str):
            return False
        relative = Path(shard)
        if not shard or relative.is_absolute() or ".." in relative.parts:
            return False
        if not (index_path.parent / relative).is_file():
            return False
    return True


def _weights_available(directory: Path, filename: str) -> bool:
    weights = directory / filename
    return weights.is_file() or _weight_index_available(weights.with_name(f"{filename}.index.json"))


def ideogram_weights_available(weights_path: Path, repository_id: str) -> bool:
    if not REPOSITORY_ID_PATTERN.fullmatch(repository_id):
        return False
    reference = (
        weights_path / "hub" / f"models--{repository_id.replace('/', '--')}" / "refs" / "main"
    )
    reference_bytes = _read_bounded(reference, MAX_SNAPSHOT_MARKER_BYTES)
    if reference_bytes is None:
        return False
    try:
        snapshot = reference_bytes.decode().strip()
    except UnicodeDecodeError:
        return False
    if SNAPSHOT_PATTERN.fullmatch(snapshot) is None:
        return False
    root = reference.parent.parent / "snapshots" / snapshot
    return (
        all(
            _json_object_available(root / item)
            for item in (
                "vae/config.json",
                "text_encoder/config.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/tokenizer.json",
            )
        )
        and all(
            _weights_available(root / component, "diffusion_pytorch_model.safetensors")
            for component in ("transformer", "unconditional_transformer", "vae")
        )
        and _weights_available(root / "text_encoder", "model.safetensors")
    )


def longcat_weights_available(weights_path: Path, revision: str) -> bool:
    marker = _read_bounded(weights_path / ".image-api-revision", MAX_SNAPSHOT_MARKER_BYTES)
    if marker is None:
        return False
    try:
        installed = marker.decode().strip()
    except UnicodeDecodeError:
        return False
    required_json = (
        "config.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/preprocessor_config.json",
        "text_processor/chat_template.json",
        "text_processor/config.json",
        "text_processor/preprocessor_config.json",
        "text_processor/special_tokens_map.json",
        "text_processor/tokenizer.json",
        "text_processor/tokenizer_config.json",
        "text_processor/vocab.json",
        "tokenizer/chat_template.json",
        "tokenizer/config.json",
        "tokenizer/preprocessor_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "vae/config.json",
    )
    return (
        installed == revision
        and all(_json_object_available(weights_path / item) for item in required_json)
        and all(
            _text_available(weights_path / item)
            for item in ("text_processor/merges.txt", "tokenizer/merges.txt")
        )
        and all(
            _weights_available(weights_path / component, filename)
            for component, filename in (
                ("text_encoder", "model.safetensors"),
                ("transformer", "diffusion_pytorch_model.safetensors"),
                ("vae", "diffusion_pytorch_model.safetensors"),
            )
        )
    )


@dataclass(frozen=True)
class Settings:
    ideogram_weights_path: Path
    longcat_edit_weights_path: Path
    longcat_edit_turbo_weights_path: Path
    longcat_edit_revision: str = LONGCAT_EDIT_REVISION
    longcat_edit_turbo_revision: str = LONGCAT_EDIT_TURBO_REVISION
    max_upload_bytes: int = 20_000_000
    max_request_bytes: int = 21_000_000
    max_input_width: int = 10_000
    max_input_height: int = 10_000
    max_input_pixels: int = 40_000_000
    max_decoded_input_bytes: int = 160_000_000
    processing_max_upload_bytes: int = 280_000_000
    processing_max_request_bytes: int = 285_000_000
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
        settings = cls(
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
            max_upload_bytes=int(os.getenv("IMAGE_API_MAX_UPLOAD_BYTES", "20000000")),
            max_request_bytes=int(os.getenv("IMAGE_API_MAX_REQUEST_BYTES", "21000000")),
            processing_max_upload_bytes=int(
                os.getenv("IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES", "280000000")
            ),
            processing_max_request_bytes=int(
                os.getenv("IMAGE_API_PROCESSING_MAX_REQUEST_BYTES", "285000000")
            ),
            magic_prompt_backend=os.getenv("IMAGE_API_MAGIC_PROMPT_BACKEND") or None,
            generation_test_mode=os.getenv("IMAGE_API_GENERATION_TEST_MODE", "false").lower()
            == "true",
        )
        return replace(
            settings,
            processing_max_native_width=int(
                os.getenv(
                    "IMAGE_API_PROCESSING_MAX_NATIVE_WIDTH", settings.processing_max_native_width
                )
            ),
            processing_max_native_height=int(
                os.getenv(
                    "IMAGE_API_PROCESSING_MAX_NATIVE_HEIGHT", settings.processing_max_native_height
                )
            ),
            processing_max_native_pixels=int(
                os.getenv(
                    "IMAGE_API_PROCESSING_MAX_NATIVE_PIXELS", settings.processing_max_native_pixels
                )
            ),
            processing_max_native_bytes=int(
                os.getenv(
                    "IMAGE_API_PROCESSING_MAX_NATIVE_BYTES", settings.processing_max_native_bytes
                )
            ),
        )

    @classmethod
    def for_tests(cls, root: Path, **overrides: Any) -> Settings:
        value = cls(
            ideogram_weights_path=root / "ideogram",
            longcat_edit_weights_path=root / "longcat",
            longcat_edit_turbo_weights_path=root / "longcat-turbo",
            max_upload_bytes=1_000_000,
            max_request_bytes=1_001_000,
            max_input_pixels=1_000_000,
            max_decoded_input_bytes=4_000_000,
            processing_max_upload_bytes=1_000_000,
            processing_max_request_bytes=1_005_000,
            processing_max_encoded_output_bytes=1_000_000,
            processing_max_input_pixels=1_000_000,
            processing_max_output_pixels=4_000_000,
            processing_max_decoded_input_bytes=4_000_000,
            processing_max_decoded_output_bytes=16_000_000,
            generation_test_mode=True,
        )
        return replace(value, **overrides)

    def admit_upscale_processing(self, width: int, height: int) -> tuple[int, int]:
        native_width, native_height = width * 4, height * 4
        native_pixels = native_width * native_height
        if (
            native_width > self.processing_max_native_width
            or native_height > self.processing_max_native_height
            or native_pixels > self.processing_max_native_pixels
            or native_pixels * 3 * 4 > self.processing_max_native_bytes
        ):
            raise ValueError("Real-ESRGAN native processing dimensions exceed configured limits")
        return native_width, native_height
