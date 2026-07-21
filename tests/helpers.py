from __future__ import annotations

from io import BytesIO

from PIL import Image


def png(mode: str = "RGB", size: tuple[int, int] = (8, 6)) -> bytes:
    color = (10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30)
    output = BytesIO()
    Image.new(mode, size, color).save(output, "PNG")
    return output.getvalue()
