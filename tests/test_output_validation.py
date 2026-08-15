from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from helpers import png
from image_api.images import ImageTooLarge, InvalidWorkerImage, validate_png_output


def test_output_validation_retains_non_dimension_contracts() -> None:
    with pytest.raises(InvalidWorkerImage, match="worker output is invalid"):
        validate_png_output(b"not an image", required_mode="RGB", max_bytes=100)

    jpeg = BytesIO()
    Image.new("RGB", (3, 2)).save(jpeg, "JPEG")
    with pytest.raises(InvalidWorkerImage, match="contract mismatch"):
        validate_png_output(jpeg.getvalue(), required_mode="RGB", max_bytes=100_000)

    with pytest.raises(InvalidWorkerImage, match="mode mismatch"):
        validate_png_output(png("RGBA", (3, 2)), required_mode="RGB", max_bytes=100_000)

    with pytest.raises(ImageTooLarge, match="encoded image exceeds configured limit"):
        validate_png_output(png("RGB", (3, 2)), required_mode="RGB", max_bytes=1)


def test_output_validation_ignores_pillow_dimension_thresholds() -> None:
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 1
        validate_png_output(png("RGB", (3, 2)), required_mode="RGB", max_bytes=100_000)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
