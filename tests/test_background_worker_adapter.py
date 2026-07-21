from __future__ import annotations

import sys
from io import BytesIO
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from helpers import png
from image_api_workers.background import app


def _install_pr7_fakes(monkeypatch, calls: list[tuple[str, dict[str, object]]]) -> None:
    package = ModuleType("rembg_api")
    package.__path__ = []  # type: ignore[attr-defined]

    birefnet = ModuleType("rembg_api.birefnet_hr")
    birefnet.DEFAULT_REVISION = "pinned"
    birefnet.BiRefNetConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    def remove_with_birefnet(data: bytes, **kwargs: object) -> bytes:
        calls.append(("birefnet", kwargs))
        return png("RGBA", (13, 7))

    birefnet.remove_with_birefnet = remove_with_birefnet

    bria = ModuleType("rembg_api.bria_rmbg")

    def remove_with_bria(data: bytes, **kwargs: object) -> bytes:
        calls.append(("bria", kwargs))
        return png("RGBA", (13, 7))

    bria.remove_with_bria_rmbg_2 = remove_with_bria

    processing = ModuleType("rembg_api.image_processing")
    processing.AlphaOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    processing.DespillOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    processing.process_png_bytes = lambda data, **kwargs: data

    monkeypatch.setitem(sys.modules, "rembg_api", package)
    monkeypatch.setitem(sys.modules, "rembg_api.birefnet_hr", birefnet)
    monkeypatch.setitem(sys.modules, "rembg_api.bria_rmbg", bria)
    monkeypatch.setitem(sys.modules, "rembg_api.image_processing", processing)


@pytest.mark.parametrize(
    ("model", "expected", "query"),
    [
        (
            "bria-rmbg-2.0",
            "bria",
            "model_input_size=1536",
        ),
        (
            "birefnet-hr-matting",
            "birefnet",
            "birefnet_inference_size=3072&birefnet_foreground_refinement=true",
        ),
    ],
)
def test_pr7_backends_dispatch_with_bounded_options_and_rgba(
    monkeypatch, tmp_path, model: str, expected: str, query: str
) -> None:
    monkeypatch.setenv("IMAGE_API_STATE_DIR", str(tmp_path))
    calls: list[tuple[str, dict[str, object]]] = []
    _install_pr7_fakes(monkeypatch, calls)
    client = TestClient(app)
    response = client.post(
        f"/internal/background-removal?model={model}&{query}",
        files={"file": ("input.png", png("RGB", (13, 7)), "image/png")},
    )
    assert response.status_code == 200
    health = client.get("/health").json()
    assert health["loaded"] is True
    assert health["loadedModel"] == model
    assert calls[0][0] == expected
    if expected == "bria":
        assert calls[0][1]["model_input_size"] == 1536
    else:
        assert calls[0][1]["inference_size"] == 3072
        assert calls[0][1]["foreground_refinement"] is True
    with Image.open(BytesIO(response.content)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.size == (13, 7)
