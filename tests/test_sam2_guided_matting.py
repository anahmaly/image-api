from __future__ import annotations

import pytest
from PIL import Image

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient
from image_api_workers import background


def test_public_background_forwards_sam2_guidance_contract(tmp_path) -> None:
    worker = FakeWorkerClient()
    settings = Settings.for_tests(tmp_path)
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=worker)
    )
    response = client.post(
        "/v1/background-removal?model=birefnet-hr-matting&sam2_guidance=true"
        "&sam2_model=sam2.1-hiera-large&sam2_mask_threshold=0.25"
        "&sam2_prompt_alpha_threshold=64&sam2_interior_erode=2"
        "&sam2_boundary_dilate=3&boundary_alpha_gamma=0.5",
        files={"file": ("input.png", png("RGB", (5, 3)), "image/png")},
    )
    assert response.status_code == 200
    assert worker.last_background["sam2_guidance"] is True
    assert worker.last_background["sam2_model"] == "sam2.1-hiera-large"
    assert worker.last_background["sam2_mask_threshold"] == 0.25
    assert worker.last_background["sam2_prompt_alpha_threshold"] == 64
    assert worker.last_background["sam2_interior_erode"] == 2
    assert worker.last_background["sam2_boundary_dilate"] == 3
    assert worker.last_background["boundary_alpha_gamma"] == 0.5


def test_guidance_fusion_preserves_hard_regions_and_gamma_unknown() -> None:
    source = Image.new("RGB", (5, 1), (9, 8, 7))
    provisional = Image.frombytes("L", (5, 1), bytes([0, 64, 128, 255, 255]))
    sam_mask = Image.frombytes("L", (5, 1), bytes([0, 255, 255, 255, 0]))

    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=0,
        boundary_dilate=0,
        boundary_alpha_gamma=0.5,
    )

    assert list(fused.getchannel("A").getdata()) == [0, 255, 255, 255, 0]
    assert fused.getpixel((0, 0))[:3] == (0, 0, 0)
    assert fused.getpixel((2, 0))[:3] == (9, 8, 7)


@pytest.mark.parametrize(
    "mask",
    [
        Image.new("L", (3, 2), 0),
        Image.new("L", (3, 2), 255),
        Image.frombytes("F", (3, 2), b"\x00" * 24),
    ],
)
def test_invalid_sam_masks_fail_explicitly(mask: Image.Image) -> None:
    with pytest.raises(ValueError, match="SAM"):
        background.validate_sam2_mask(mask, (3, 2))


def test_guidance_off_does_not_create_sam_adapter(monkeypatch) -> None:
    monkeypatch.setattr(background, "_sam2_adapter", None)
    created = False

    def fail_factory() -> object:
        nonlocal created
        created = True
        raise AssertionError("SAM must not load")

    monkeypatch.setattr(background, "_create_sam2_adapter", fail_factory)
    assert background._sam2_adapter is None
    assert created is False
