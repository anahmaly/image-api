from __future__ import annotations

from io import BytesIO
from typing import Protocol

import httpx
from PIL import Image


class WorkerUnavailable(RuntimeError):
    pass


class WorkerClient(Protocol):
    model_invocations: int
    model_loads: int

    def health(self) -> dict[str, dict[str, object]]: ...
    def upscale(self, data: bytes, **parameters: object) -> bytes: ...
    def background(self, data: bytes, **parameters: object) -> bytes: ...


class HttpWorkerClient:
    model_invocations = 0
    model_loads = 0

    def __init__(
        self,
        upscale_url: str,
        background_url: str,
        timeout_seconds: float,
        max_output_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.upscale_url = upscale_url.rstrip("/")
        self.background_url = background_url.rstrip("/")
        self.max_output_bytes = max_output_bytes
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_seconds), transport=transport)

    def _get_health(self, base: str) -> dict[str, object]:
        try:
            response = httpx.get(f"{base}/health", timeout=0.25)
            response.raise_for_status()
            body = response.json()
            return {
                "ready": bool(body.get("ready", False)),
                "loaded": bool(body.get("loaded", False)),
                "device": body.get("device", "unavailable"),
            }
        except Exception:
            return {"ready": False, "loaded": False, "device": "unavailable"}

    def health(self) -> dict[str, dict[str, object]]:
        return {
            "upscale": self._get_health(self.upscale_url),
            "background-removal": self._get_health(self.background_url),
        }

    def _post(self, url: str, data: bytes, parameters: dict[str, object]) -> bytes:
        try:
            with self.client.stream(
                "POST",
                url,
                params={
                    key: None if value is None else str(value) for key, value in parameters.items()
                },
                files={"file": ("input", data, "application/octet-stream")},
            ) as response:
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit():
                    if int(declared) > self.max_output_bytes:
                        raise WorkerUnavailable("worker output exceeds configured limit")
                output = bytearray()
                for chunk in response.iter_bytes():
                    if len(chunk) > self.max_output_bytes - len(output):
                        raise WorkerUnavailable("worker output exceeds configured limit")
                    output.extend(chunk)
                return bytes(output)
        except Exception as exc:
            raise WorkerUnavailable("worker request failed") from exc

    def upscale(self, data: bytes, **parameters: object) -> bytes:
        return self._post(f"{self.upscale_url}/internal/upscale", data, parameters)

    def background(self, data: bytes, **parameters: object) -> bytes:
        return self._post(f"{self.background_url}/internal/background-removal", data, parameters)


class FakeWorkerClient:
    def __init__(self) -> None:
        self.model_invocations = 0
        self.model_loads = 0
        self.last_upscale: dict[str, object] = {}
        self.last_background: dict[str, object] = {}

    def health(self) -> dict[str, dict[str, object]]:
        return {
            "upscale": {"ready": True, "loaded": False, "device": "cpu-test"},
            "background-removal": {"ready": True, "loaded": False, "device": "cpu-test"},
        }

    @staticmethod
    def _open(data: bytes) -> Image.Image:
        with Image.open(BytesIO(data)) as image:
            return image.copy()

    def upscale(self, data: bytes, **parameters: object) -> bytes:
        self.model_invocations += 1
        self.last_upscale = parameters
        image = self._open(data).convert("RGBA" if self._open(data).mode == "RGBA" else "RGB")
        scale_value = parameters["outscale"]
        if not isinstance(scale_value, (int, float)):
            raise ValueError("fake worker outscale must be numeric")
        scale = float(scale_value)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        output = BytesIO()
        image.save(output, "PNG")
        return output.getvalue()

    def background(self, data: bytes, **parameters: object) -> bytes:
        self.model_invocations += 1
        self.last_background = parameters
        image = self._open(data).convert("RGBA")
        output = BytesIO()
        image.save(output, "PNG")
        return output.getvalue()
