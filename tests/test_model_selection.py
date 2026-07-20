import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
MAIN_PATH = APP_DIR / "main.py"
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "compose.yml").read_text()
README = (ROOT / "README.md").read_text()


def _load_model_config():
    path = APP_DIR / "model_config.py"
    spec = importlib.util.spec_from_file_location("model_config", str(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unset_and_blank_resolve_to_standard_23_block_spec():
    config = _load_model_config()

    for value in (None, "", "   "):
        selected = config.resolve_model(value)
        assert selected.canonical_name == "RealESRGAN_x4plus"
        assert selected.num_block == 23
        assert selected.network_scale == 4
        assert selected.model_path == "/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"


def test_canonical_standard_model_resolves_exactly():
    config = _load_model_config()

    selected = config.resolve_model("RealESRGAN_x4plus")

    assert selected.canonical_name == "RealESRGAN_x4plus"
    assert selected.num_block == 23


def test_canonical_anime_model_resolves_exact_architecture_path_and_scale():
    config = _load_model_config()

    selected = config.resolve_model("RealESRGAN_x4plus_anime_6B")

    assert selected.canonical_name == "RealESRGAN_x4plus_anime_6B"
    assert selected.num_block == 6
    assert selected.network_scale == 4
    assert selected.model_path == (
        "/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth"
    )


def test_friendly_exact_alias_resolves_to_anime_model():
    config = _load_model_config()

    selected = config.resolve_model("  real-esrgan anime 6b  ")

    assert selected.canonical_name == "RealESRGAN_x4plus_anime_6B"


def test_unsupported_model_fails_with_both_supported_names_and_no_fallback():
    config = _load_model_config()

    with pytest.raises(ValueError) as error:
        config.resolve_model("anime")

    message = str(error.value)
    assert "Unsupported REALESRGAN_MODEL 'anime'" in message
    assert (
        "Supported models: RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B"
        in message
    )


def test_model_specs_are_immutable():
    config = _load_model_config()
    selected = config.resolve_model(None)

    with pytest.raises(AttributeError):
        selected.num_block = 6


def test_main_constructs_runtime_from_selected_model_spec():
    source = MAIN_PATH.read_text()
    tree = ast.parse(source)

    assert 'os.getenv("REALESRGAN_MODEL")' in source
    assert "resolve_model(" in source
    assert "num_block=selected_model.num_block" in source
    assert "scale=selected_model.network_scale" in source
    assert "model_path=selected_model.model_path" in source
    assert "scale=selected_model.network_scale" in source
    assert "num_block=23" not in source
    assert 'model_path = "/Real-ESRGAN/weights/' not in source

    upsampler_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RealESRGANer"
    )
    keywords = {keyword.arg: keyword.value for keyword in upsampler_call.keywords}
    assert isinstance(keywords["scale"], ast.Attribute)
    assert keywords["scale"].attr == "network_scale"
    assert isinstance(keywords["model_path"], ast.Attribute)
    assert keywords["model_path"].attr == "model_path"


def test_main_preserves_default_tile_lock_and_finally_restore_contract():
    source = MAIN_PATH.read_text()

    assert "DEFAULT_TILE = 512" in source
    assert "tile=DEFAULT_TILE" in source
    assert "upsampler_lock = asyncio.Lock()" in source
    assert "async with upsampler_lock:" in source
    assert "previous_tile_size = upsampler.tile_size" in source
    assert "upsampler.tile_size = tile" in source
    assert "finally:" in source
    assert "upsampler.tile_size = previous_tile_size" in source


def test_health_route_reports_canonical_model_and_device_without_path():
    tree = ast.parse(MAIN_PATH.read_text())
    health = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "health"
    )

    decorator = health.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert isinstance(decorator.args[0], ast.Constant)
    assert decorator.args[0].value == "/health"
    returned = next(node.value for node in health.body if isinstance(node, ast.Return))
    assert isinstance(returned, ast.Dict)
    assert all(isinstance(key, ast.Constant) for key in returned.keys)
    keys = [
        key.value for key in returned.keys if isinstance(key, ast.Constant)
    ]
    values = dict(zip(keys, returned.values))
    assert isinstance(values["status"], ast.Constant)
    assert values["status"].value == "ok"
    assert isinstance(values["model"], ast.Attribute)
    assert values["model"].attr == "canonical_name"
    assert isinstance(values["device"], ast.Attribute)
    assert values["device"].attr == "type"
    assert "path" not in keys


def test_dockerfile_embeds_both_official_weights_at_exact_paths():
    assert (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth"
    ) in DOCKERFILE
    assert (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
        "RealESRGAN_x4plus_anime_6B.pth"
    ) in DOCKERFILE
    assert "-O /Real-ESRGAN/weights/RealESRGAN_x4plus.pth" in DOCKERFILE
    assert (
        "-O /Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth" in DOCKERFILE
    )


def test_compose_passes_model_selection_with_standard_default():
    assert "REALESRGAN_MODEL: ${REALESRGAN_MODEL:-RealESRGAN_x4plus}" in COMPOSE


def test_readme_documents_exact_startup_selection_contract():
    assert "REALESRGAN_MODEL" in README
    assert "RealESRGAN_x4plus" in README
    assert "RealESRGAN_x4plus_anime_6B" in README
    assert "Real-ESRGAN Anime 6B" in README
    assert "default" in README.lower()
    assert "REALESRGAN_MODEL=RealESRGAN_x4plus_anime_6B docker compose" in README
    assert (
        "docker run -e REALESRGAN_MODEL=RealESRGAN_x4plus_anime_6B" in README
    )
    assert "rebuild" in README.lower()
    assert '"status": "ok"' in README
    assert '"model": "RealESRGAN_x4plus_anime_6B"' in README
    assert '"device": "cuda"' in README
