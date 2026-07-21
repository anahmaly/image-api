from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


REPOSITORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SNAPSHOT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


def _weight_index_available(index_path: Path) -> bool:
    try:
        if index_path.stat().st_size > 5_000_000:
            return False
        with index_path.open() as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards: set[str] = set()
    for shard in weight_map.values():
        if not isinstance(shard, str):
            return False
        shards.add(shard)
    for shard in shards:
        relative = Path(shard)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        if not (index_path.parent / relative).is_file():
            return False
    return True


def _weights_available(directory: Path, filename: str) -> bool:
    weights = directory / filename
    return weights.is_file() or _weight_index_available(
        weights.with_name(f"{weights.name}.index.json")
    )


def ideogram_weights_available(weights_path: Path, repository_id: str) -> bool:
    if not REPOSITORY_ID_PATTERN.fullmatch(repository_id):
        return False
    repository_cache = weights_path / "hub" / f"models--{repository_id.replace('/', '--')}"
    reference = repository_cache / "refs" / "main"
    try:
        with reference.open() as handle:
            snapshot_name = handle.read(65).strip()
    except OSError:
        return False
    if not SNAPSHOT_PATTERN.fullmatch(snapshot_name):
        return False
    snapshot = repository_cache / "snapshots" / snapshot_name
    diffusion_components = (snapshot / "transformer", snapshot / "unconditional_transformer")
    required = (
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
    )
    return (
        all((snapshot / relative).is_file() for relative in required)
        and all(
            _weights_available(component, "diffusion_pytorch_model.safetensors")
            for component in diffusion_components
        )
        and _weights_available(snapshot / "text_encoder", "model.safetensors")
    )


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    database_path: Path
    output_dir: Path
    gpu_lane_path: Path
    generation_heartbeat_path: Path
    ideogram_weights_path: Path
    max_request_bytes: int = 21_000_000
    max_upload_bytes: int = 20_000_000
    max_input_width: int = 10_000
    max_input_height: int = 10_000
    max_input_pixels: int = 40_000_000
    max_output_pixels: int = 80_000_000
    max_queue_depth: int = 100
    worker_timeout_seconds: float = 120.0
    lane_timeout_seconds: float = 2.0
    generation_heartbeat_max_age_seconds: float = 15.0
    magic_prompt_backend: str | None = None
    cuda_available: bool = False
    generation_test_mode: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        state = Path(os.getenv("IMAGE_API_STATE_DIR", "/state"))
        values = cls(
            state_dir=state,
            database_path=state / "tasks.sqlite3",
            output_dir=state / "outputs",
            gpu_lane_path=state / "gpu-lane.lock",
            generation_heartbeat_path=state / "generation-worker.heartbeat",
            ideogram_weights_path=Path(
                os.getenv("IMAGE_API_IDEOGRAM_WEIGHTS_PATH", "/models/ideogram-4-nf4")
            ),
            max_request_bytes=int(os.getenv("IMAGE_API_MAX_REQUEST_BYTES", "21000000")),
            max_upload_bytes=int(os.getenv("IMAGE_API_MAX_UPLOAD_BYTES", "20000000")),
            max_input_width=int(os.getenv("IMAGE_API_MAX_INPUT_WIDTH", "10000")),
            max_input_height=int(os.getenv("IMAGE_API_MAX_INPUT_HEIGHT", "10000")),
            max_input_pixels=int(os.getenv("IMAGE_API_MAX_INPUT_PIXELS", "40000000")),
            max_output_pixels=int(os.getenv("IMAGE_API_MAX_OUTPUT_PIXELS", "80000000")),
            max_queue_depth=int(os.getenv("IMAGE_API_MAX_QUEUE_DEPTH", "100")),
            worker_timeout_seconds=float(os.getenv("IMAGE_API_WORKER_TIMEOUT_SECONDS", "120")),
            lane_timeout_seconds=float(os.getenv("IMAGE_API_LANE_TIMEOUT_SECONDS", "2")),
            generation_heartbeat_max_age_seconds=float(
                os.getenv("IMAGE_API_GENERATION_HEARTBEAT_MAX_AGE_SECONDS", "15")
            ),
            magic_prompt_backend=os.getenv("IMAGE_API_MAGIC_PROMPT_BACKEND") or None,
            cuda_available=_bool("IMAGE_API_CUDA_AVAILABLE", False),
            generation_test_mode=_bool("IMAGE_API_GENERATION_TEST_MODE", False),
        )
        values.validate()
        return values

    def validate(self) -> None:
        positive = {
            "IMAGE_API_MAX_REQUEST_BYTES": self.max_request_bytes,
            "IMAGE_API_MAX_UPLOAD_BYTES": self.max_upload_bytes,
            "IMAGE_API_MAX_INPUT_WIDTH": self.max_input_width,
            "IMAGE_API_MAX_INPUT_HEIGHT": self.max_input_height,
            "IMAGE_API_MAX_INPUT_PIXELS": self.max_input_pixels,
            "IMAGE_API_MAX_OUTPUT_PIXELS": self.max_output_pixels,
            "IMAGE_API_MAX_QUEUE_DEPTH": self.max_queue_depth,
        }
        if any(value < 1 for value in positive.values()):
            raise ValueError("image-api limits must be positive")
        if self.max_request_bytes < self.max_upload_bytes:
            raise ValueError("IMAGE_API_MAX_REQUEST_BYTES must cover IMAGE_API_MAX_UPLOAD_BYTES")
        if (
            self.worker_timeout_seconds <= 0
            or self.lane_timeout_seconds <= 0
            or self.generation_heartbeat_max_age_seconds <= 0
        ):
            raise ValueError("image-api timeouts must be positive")

    @classmethod
    def for_tests(cls, root: Path, **overrides: Any) -> "Settings":
        base = cls(
            state_dir=root,
            database_path=root / "tasks.sqlite3",
            output_dir=root / "outputs",
            gpu_lane_path=root / "gpu-lane.lock",
            generation_heartbeat_path=root / "generation-worker.heartbeat",
            ideogram_weights_path=root / "ideogram-weights",
            max_upload_bytes=1_000_000,
            max_request_bytes=1_100_000,
            max_input_pixels=1_000_000,
            max_output_pixels=4_000_000,
            cuda_available=True,
            generation_test_mode=True,
        )
        value = replace(base, **overrides)
        value.validate()
        return value
