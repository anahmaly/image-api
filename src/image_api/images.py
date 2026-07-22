from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from math import isfinite
from typing import IO

from PIL import Image, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


class ImageTooLarge(ValueError):
    pass


class InvalidWorkerImage(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str


def _decoded_bytes(width: int, height: int, mode: str) -> int:
    _ = Image.getmodebands(mode)  # Reject unknown Pillow modes while budgeting conservatively.
    return width * height * 4


def validate_dimensions(
    width: int,
    height: int,
    *,
    max_width: int,
    max_height: int,
    max_pixels: int,
    max_decoded_bytes: int | None,
) -> None:
    if width < 1 or height < 1:
        raise InvalidImage("image dimensions are invalid")
    pixels = width * height
    if width > max_width or height > max_height or pixels > max_pixels:
        raise ImageTooLarge("image dimensions exceed configured limits")
    if max_decoded_bytes is not None and pixels * 4 > max_decoded_bytes:
        raise ImageTooLarge("decoded image exceeds configured limit")


def processing_output_size(info: ImageInfo, outscale: float) -> tuple[int, int]:
    if not isfinite(outscale) or not 1 <= outscale <= 4:
        raise ValueError("outscale must be between one and four")
    return (round(info.width * outscale), round(info.height * outscale))


def validate_image(
    data: bytes | IO[bytes],
    *,
    max_bytes: int,
    max_width: int,
    max_height: int,
    max_pixels: int,
    max_decoded_bytes: int | None = None,
    worker_output: bool = False,
) -> ImageInfo:
    invalid_type = InvalidWorkerImage if worker_output else InvalidImage
    stream: IO[bytes]
    original_position = 0
    if isinstance(data, bytes):
        if not data:
            raise invalid_type("image is empty")
        if len(data) > max_bytes:
            raise ImageTooLarge("encoded image exceeds configured limit")
        stream = BytesIO(data)
    else:
        stream = data
        original_position = stream.tell()
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(0)
        if size < 1:
            raise invalid_type("image is empty")
        if size > max_bytes:
            raise ImageTooLarge("encoded image exceeds configured limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(stream) as image:
                width, height = image.size
                if width < 1 or height < 1:
                    raise invalid_type("image dimensions are invalid")
                if width > max_width or height > max_height or width * height > max_pixels:
                    raise ImageTooLarge("image dimensions exceed configured limits")
                if (
                    max_decoded_bytes is not None
                    and _decoded_bytes(width, height, image.mode) > max_decoded_bytes
                ):
                    raise ImageTooLarge("decoded image exceeds configured limit")
                image.verify()
            stream.seek(0)
            with Image.open(stream) as image:
                image.load()
                return ImageInfo(width, height, image.mode)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageTooLarge("image dimensions exceed configured limits") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, (ImageTooLarge, InvalidImage, InvalidWorkerImage)):
            raise
        raise invalid_type("bytes are not a valid image") from exc
    finally:
        if not isinstance(data, bytes):
            stream.seek(original_position)


def validate_png_output(
    data: bytes | IO[bytes],
    *,
    expected_size: tuple[int, int] | None,
    required_mode: str | None,
    max_bytes: int,
    max_pixels: int,
    max_decoded_bytes: int | None = None,
) -> None:
    maximum_size = expected_size or (max_pixels, max_pixels)
    info = validate_image(
        data,
        max_bytes=max_bytes,
        max_width=maximum_size[0],
        max_height=maximum_size[1],
        max_pixels=max_pixels,
        max_decoded_bytes=max_decoded_bytes,
        worker_output=True,
    )
    stream = BytesIO(data) if isinstance(data, bytes) else data
    position = stream.tell()
    try:
        stream.seek(0)
        with Image.open(stream) as image:
            image_format = image.format
    except Exception as exc:
        raise InvalidWorkerImage("worker output is invalid") from exc
    finally:
        stream.seek(position)
    if image_format != "PNG" or (
        expected_size is not None and (info.width, info.height) != expected_size
    ):
        raise InvalidWorkerImage("worker output contract mismatch")
    if required_mode is not None and info.mode != required_mode:
        raise InvalidWorkerImage("worker output mode mismatch")
