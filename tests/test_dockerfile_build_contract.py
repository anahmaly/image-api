from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "compose.yml").read_text()
MAIN = (ROOT / "app" / "main.py").read_text()


def _realesrgan_dependency_install_block():
    marker = "# 6. Install Real-ESRGAN deps"
    start = DOCKERFILE.index(marker)
    end = DOCKERFILE.index("# 7. Copy API code", start)
    return DOCKERFILE[start:end]


def test_lmdb_build_toolchain_is_transient_and_ordered_around_source_installs():
    install_block = _realesrgan_dependency_install_block()

    apt_update = install_block.index("apt-get update")
    toolchain_install = install_block.index(
        "apt-get install -y --no-install-recommends build-essential"
    )
    first_lmdb_capable_pip = install_block.index(
        "pip install --no-cache-dir basicsr==1.4.2"
    )
    setup_develop = install_block.index("python setup.py develop")
    toolchain_purge = install_block.index(
        "apt-get purge -y --auto-remove build-essential"
    )
    apt_lists_cleanup = install_block.index("rm -rf /var/lib/apt/lists/*")

    assert apt_update < toolchain_install < first_lmdb_capable_pip
    assert first_lmdb_capable_pip < setup_develop < toolchain_purge
    assert toolchain_purge < apt_lists_cleanup
    assert "python setup.py develop && \\\n    apt-get purge" in install_block


def test_official_model_weight_urls_and_destinations_are_unchanged():
    assert (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth"
    ) in DOCKERFILE
    assert "-O /Real-ESRGAN/weights/RealESRGAN_x4plus.pth" in DOCKERFILE
    assert (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
        "RealESRGAN_x4plus_anime_6B.pth"
    ) in DOCKERFILE
    assert (
        "-O /Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth" in DOCKERFILE
    )


def test_gpu_runtime_and_upscale_selection_contracts_are_unchanged():
    assert "FROM python:3.8-slim-bullseye" in DOCKERFILE
    assert "torch==2.1.0+cu118" in DOCKERFILE
    assert "torchvision==0.16.0+cu118" in DOCKERFILE
    assert "torchaudio==2.1.0+cu118" in DOCKERFILE
    assert "--index-url https://download.pytorch.org/whl/cu118" in DOCKERFILE
    assert "EXPOSE 8000" in DOCKERFILE
    assert (
        'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]'
        in DOCKERFILE
    )

    assert '"8000:8000"' in COMPOSE
    assert "REALESRGAN_MODEL: ${REALESRGAN_MODEL:-RealESRGAN_x4plus}" in COMPOSE
    assert "capabilities: [gpu]" in COMPOSE
    assert "gpus: all" in COMPOSE

    assert "outscale: float = 2.0" in MAIN
    assert "tile: int = Query(" in MAIN
    assert 'os.getenv("REALESRGAN_MODEL")' in MAIN
