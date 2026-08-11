from __future__ import annotations

import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repository_text() -> str:
    parts = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or ".venv" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        if (
            "tests" in path.parts
            or path.name == "test_cutover_contract.py"
            or "licenses" in path.parts
            or path.name in {"NOTICE.md", "README.md"}
        ):
            continue
        if path.suffix in {".pyc", ".png"} or path.name == "uv.lock":
            continue
        try:
            parts.append(f"{path.relative_to(ROOT)}\n{path.read_text()}")
        except UnicodeDecodeError:
            pass
    return "\n".join(parts)


def test_old_public_identity_and_route_contract_are_absent() -> None:
    text = repository_text()
    assert "real-esrgan-api" not in text.lower()
    assert '"/upscale/"' not in text
    assert "REALESRGAN_MODEL" not in text
    assert "REALESRGAN_" not in text
    assert "/models/realesrgan" not in text
    compose = (ROOT / "compose.yml").read_text()
    assert "container_name: realesrgan" not in compose
    assert "\n  realesrgan:" not in compose
    assert "image-api" in compose


def test_only_gateway_publishes_ports() -> None:
    compose = (ROOT / "compose.yml").read_text().splitlines()
    port_lines = [line for line in compose if line.strip() == "ports:"]
    assert len(port_lines) == 1


def test_production_has_no_durable_database_or_state_volume_path() -> None:
    text = repository_text()
    for forbidden in (
        "TaskStore",
        "tasks.sqlite",
        "sqlite3",
        "state-init",
        "IMAGE_API_STATE",
        "IMAGE_API_ENABLE_PROCESSING_RUNNER",
        "/state",
    ):
        assert forbidden not in text


def test_compose_has_no_legacy_background_model_mount_contract() -> None:
    compose = (ROOT / "compose.yml").read_text()
    assert "/models/rembg" not in compose
    assert "IMAGE_API_REMBG_MODELS_PATH" not in compose
    assert "IMAGE_API_REMBG_WEIGHTS_PATH" not in compose
    assert "IMAGE_API_REMBG_WEIGHTS_HOST_PATH" not in compose


def test_compose_excludes_all_internal_peers_from_ambient_proxies() -> None:
    compose = (ROOT / "compose.yml").read_text()
    exclusions = "localhost,127.0.0.1,::1,upscale-worker,background-worker,generation-worker"
    for service in ("image-api", "upscale-worker", "background-worker", "generation-worker"):
        match = re.search(rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  \S|\Z)", compose)
        assert match is not None
        body = match.group("body")
        assert f"NO_PROXY: {exclusions}" in body
        assert f"no_proxy: {exclusions}" in body


def test_gateway_compose_keeps_finite_request_and_file_limits_distinct() -> None:
    compose = (ROOT / "compose.yml").read_text()
    for name, value in (
        ("IMAGE_API_MAX_UPLOAD_BYTES", "20000000"),
        ("IMAGE_API_MAX_REQUEST_BYTES", "21000000"),
        ("IMAGE_API_PROCESSING_MAX_UPLOAD_BYTES", "280000000"),
        ("IMAGE_API_PROCESSING_MAX_REQUEST_BYTES", "285000000"),
    ):
        assert f"{name}: ${{{name}:-{value}}}" in compose


def test_generation_compose_healthcheck_requires_responsive_worker() -> None:
    compose = (ROOT / "compose.yml").read_text()
    match = re.search(r"(?ms)^  generation-worker:\n(?P<body>.*?)(?=^  \S|\Z)", compose)
    assert match is not None
    healthcheck = match.group("body")
    assert "urllib.request.urlopen('http://127.0.0.1:9003/health', timeout=3)" in healthcheck
    assert "json.load" not in healthcheck
    assert "['ready']" not in healthcheck
    assert "restart: on-failure" in match.group("body")


def test_generation_compose_mounts_the_existing_physical_model_roots() -> None:
    compose = (ROOT / "compose.yml").read_text()
    match = re.search(r"(?ms)^  generation-worker:\n(?P<body>.*?)(?=^  \S|\Z)", compose)
    assert match is not None
    worker = match.group("body")
    assert "${IMAGE_API_MODELS_HOST_PATH:-./models}:/models:ro" in worker
    assert "IMAGE_API_IDEOGRAM_WEIGHTS_PATH: /models/ideogram-4-nf4" in worker
    assert "IMAGE_API_LONGCAT_EDIT_WEIGHTS_PATH: /models/longcat-image-edit" in worker
    assert "IMAGE_API_LONGCAT_EDIT_TURBO_WEIGHTS_PATH: /models/longcat-image-edit-turbo" in worker


def test_gateway_compose_healthcheck_accepts_truthful_partial_gateway_readiness() -> None:
    compose = (ROOT / "compose.yml").read_text()
    match = re.search(r"(?ms)^  image-api:\n(?P<body>.*?)(?=^  \S|\Z)", compose)
    assert match is not None
    healthcheck = match.group("body")
    assert "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" in healthcheck
    assert "['service'] == 'image-api'" in healthcheck
    assert "['status'] in ('ok', 'degraded')" in healthcheck


def test_pinned_upstream_sources_and_no_weight_download_commands() -> None:
    text = repository_text()
    assert "dd7b6fd434cff2077ce6e9a0cab46fe254f26f1f" in text
    assert "990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2" in text
    dockerfiles = "\n".join(path.read_text() for path in ROOT.glob("Dockerfile*"))
    assert "wget" not in dockerfiles
    assert "curl" not in dockerfiles
    assert "hf auth" not in dockerfiles.lower()


def test_background_install_pins_birefnet_runtime_dependency() -> None:
    dockerfiles = list(ROOT.glob("Dockerfile*"))
    background = (ROOT / "Dockerfile.background").read_text().replace("\\\n", " ")
    install_line = next(
        line.removeprefix("RUN ")
        for line in background.splitlines()
        if line.startswith("RUN ") and "rembg-api.git" in line
    )
    tokens = shlex.split(install_line)

    einops_tokens = [token for token in tokens if token.lower().startswith("einops")]
    assert einops_tokens == ["einops==0.8.2"]
    assert (
        "git+https://github.com/anahmaly/rembg-api.git@dd7b6fd434cff2077ce6e9a0cab46fe254f26f1f"
    ) in tokens
    assert "--break-system-packages" not in background
    assert all(
        "einops" not in path.read_text().lower()
        for path in dockerfiles
        if path.name != "Dockerfile.background"
    )


def test_background_sam2_install_uses_verified_source_without_cuda_build_isolation() -> None:
    background = (ROOT / "Dockerfile.background").read_text().replace("\\\n", " ")
    sam2_install = next(
        line.removeprefix("RUN ")
        for line in background.splitlines()
        if line.startswith("RUN ") and "facebookresearch/sam2.git" in line
    )
    tokens = shlex.split(sam2_install)

    assert (
        "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
    ) in tokens
    assert "SAM2_BUILD_CUDA=0" in tokens
    assert "--no-build-isolation" in tokens


def test_generation_install_handles_pep_668_and_keeps_ideogram_pinned() -> None:
    dockerfiles = list(ROOT.glob("Dockerfile*"))
    generation = (ROOT / "Dockerfile.generation").read_text().replace("\\\n", " ")
    install_line = next(
        line.removeprefix("RUN ")
        for line in generation.splitlines()
        if line.startswith("RUN ") and "ideogram4.git" in line
    )
    tokens = shlex.split(install_line)

    assert tokens[:4] == ["python", "-m", "pip", "install"]
    assert "--break-system-packages" in tokens
    assert "transformers==4.56.2" in tokens
    assert "protobuf==6.33.4" in tokens
    assert (
        "git+https://github.com/ideogram-oss/ideogram4.git@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2"
    ) in tokens
    assert all(
        "--break-system-packages" not in path.read_text()
        for path in dockerfiles
        if path.name != "Dockerfile.generation"
    )
