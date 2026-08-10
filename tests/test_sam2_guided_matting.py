from __future__ import annotations

import struct

import pytest
from PIL import Image

from image_api_workers import background


def test_guidance_preserves_enclosed_opening_wider_than_closing_kernel() -> None:
    source = Image.new("RGB", (15, 15), (9, 8, 7))
    provisional = Image.new("L", (15, 15), 0)
    sam_mask = Image.new("L", (15, 15), 0)
    for x in range(3, 12):
        sam_mask.putpixel((x, 3), 255)
        sam_mask.putpixel((x, 11), 255)
    for y in range(3, 12):
        sam_mask.putpixel((3, y), 255)
        sam_mask.putpixel((11, y), 255)
    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=0,
        boundary_dilate=1,
        boundary_alpha_gamma=1.0,
        provisional_foreground_threshold=128,
    )
    alpha = fused.getchannel("A")
    assert alpha.getpixel((3, 7)) == 255
    assert alpha.getpixel((5, 5)) == 0


@pytest.mark.parametrize("value", [float("nan"), 1.1, 1.0, 0.0])
def test_invalid_sam_masks_fail_explicitly(value: float) -> None:
    probability = Image.frombytes("F", (2, 2), struct.pack("4f", value, value, value, value))
    with pytest.raises(ValueError, match="SAM"):
        background._binary_sam2_mask(probability, threshold=0.5, size=(2, 2))


def test_binary_sam2_mask_accepts_ordinary_probabilities() -> None:
    probability = Image.frombytes("F", (2, 1), struct.pack("2f", 0.1, 0.9))
    assert list(
        background._binary_sam2_mask(probability, threshold=0.5, size=(2, 1)).getdata()
    ) == [0, 255]
