from __future__ import annotations

import logging
from io import BytesIO
from tempfile import SpooledTemporaryFile
from typing import Callable, IO, Protocol, TypeAlias

import httpx
from PIL import Image

logger = logging.getLogger(__name__)
WorkerInput: TypeAlias = bytes | IO[bytes]
WorkerOutput: TypeAlias = bytes | IO[bytes]
OUTPUT_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024


class WorkerUnavailable(RuntimeError):
    pass


class SanitizedPeerFailure(RuntimeError):
    """A bounded peer failure suitable for exception-object logging and propagation."""

    def __init__(self, category: str, status_code: int | None = None) -> None:
        self.category = category
        self.status_code = status_code
        detail = f"peer {category} failure"
        if status_code is not None:
            detail += f" status={status_code}"
        super().__init__(detail)


def _sanitize_peer_failure(exc: Exception) -> SanitizedPeerFailure:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return SanitizedPeerFailure(
            "http_status", status_code if 100 <= status_code <= 599 else None
        )
    if isinstance(exc, httpx.HTTPError):
        return SanitizedPeerFailure("transport")
    if isinstance(exc, WorkerUnavailable):
        return SanitizedPeerFailure("worker_unavailable")
    return SanitizedPeerFailure("unexpected")


def _log_sanitized_peer_failure(message: str, failure: SanitizedPeerFailure, *args: object) -> None:
    """Log a fresh safe traceback after the raw peer exception's scope has ended."""
    try:
        raise failure from None
    except SanitizedPeerFailure as safe_failure:
        logger.error(
            message, *args, exc_info=(type(safe_failure), safe_failure, safe_failure.__traceback__)
        )


def _raise_worker_unavailable(message: str, failure: SanitizedPeerFailure) -> None:
    """Raise with a safe exception chain and no raw HTTP/client exception context."""
    try:
        raise failure from None
    except SanitizedPeerFailure as safe_failure:
        raise WorkerUnavailable(message) from safe_failure


def create_internal_http_client(
    timeout: httpx.Timeout | float,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Create a client for Docker-internal peer traffic without ambient proxy authority."""
    return httpx.Client(timeout=timeout, transport=transport, trust_env=False)


class WorkerClient(Protocol):
    model_invocations: int
    model_loads: int

    def health(self) -> dict[str, dict[str, object]]: ...
    def upscale(self, data: WorkerInput, **parameters: object) -> WorkerOutput: ...
    def background(self, data: WorkerInput, **parameters: object) -> WorkerOutput: ...
    def generation(self, request: dict[str, object]) -> WorkerOutput: ...
    def image_edit(self, data: WorkerInput, **parameters: object) -> WorkerOutput: ...
    def unload_all(self) -> dict[str, dict[str, object]]: ...


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
        generation_url: str = "http://generation-worker:9003",
    ) -> None:
        self.urls = {
            "upscale": upscale_url.rstrip("/"),
            "background-removal": background_url.rstrip("/"),
            "generation": generation_url.rstrip("/"),
        }
        self.upscale_url = self.urls["upscale"]
        self.background_url = self.urls["background-removal"]
        self.max_output_bytes = max_output_bytes
        self.client = create_internal_http_client(httpx.Timeout(timeout_seconds), transport)

    def _get_health(self, capability: str, base: str) -> dict[str, object]:
        failure: SanitizedPeerFailure | None = None
        try:
            response = self.client.get(f"{base}/health", timeout=0.25)
            response.raise_for_status()
            body = response.json()
            raw_device = body.get("device")
            device = (
                raw_device if raw_device in {"cuda", "cpu-test", "unavailable"} else "unavailable"
            )
            result: dict[str, object] = {
                "ready": bool(body.get("ready")),
                "loaded": bool(body.get("loaded")),
                "device": device,
                "workerReachable": True,
            }
            loaded_model = body.get("loadedModel")
            if isinstance(loaded_model, str):
                result["loadedModel"] = loaded_model
            if "weightsAvailable" in body:
                result["weightsAvailable"] = bool(body["weightsAvailable"])
            models = body.get("models")
            if isinstance(models, dict):
                allowed_models = {
                    "ideogram-4-nf4",
                    "longcat-image-edit",
                    "longcat-image-edit-turbo",
                }
                result["models"] = {
                    name: {
                        "weightsAvailable": bool(value.get("weightsAvailable", False)),
                        "loaded": bool(value.get("loaded", False)),
                    }
                    for name, value in models.items()
                    if name in allowed_models and isinstance(value, dict)
                }
            return result
        except Exception as exc:
            failure = _sanitize_peer_failure(exc)
        if failure is not None:
            _log_sanitized_peer_failure(
                "worker health check failed: capability=%s", failure, capability
            )
        return {"ready": False, "loaded": False, "device": "unavailable"}

    def health(self) -> dict[str, dict[str, object]]:
        return {name: self._get_health(name, url) for name, url in self.urls.items()}

    def unload_all(self) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        for name, base in self.urls.items():
            failure: SanitizedPeerFailure | None = None
            try:
                response = self.client.post(f"{base}/internal/unload")
                response.raise_for_status()
                body = response.json()
                results[name] = {"unloaded": bool(body.get("unloaded", False))}
            except Exception as exc:
                failure = _sanitize_peer_failure(exc)
            if failure is not None:
                _log_sanitized_peer_failure("worker unload failed: capability=%s", failure, name)
                results[name] = {"unloaded": False, "error": "worker_unavailable"}
        return results

    def _post(self, url: str, data: WorkerInput, parameters: dict[str, object]) -> IO[bytes]:
        output: IO[bytes] | None = None
        failure: SanitizedPeerFailure | None = None
        try:
            if not isinstance(data, bytes):
                data.seek(0)
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
                output = SpooledTemporaryFile(max_size=OUTPUT_SPOOL_MEMORY_BYTES, mode="w+b")
                total = 0
                for chunk in response.iter_bytes():
                    if len(chunk) > self.max_output_bytes - total:
                        raise WorkerUnavailable("worker output exceeds configured limit")
                    output.write(chunk)
                    total += len(chunk)
                output.seek(0)
                return output
        except Exception as exc:
            if output is not None:
                output.close()
            failure = _sanitize_peer_failure(exc)
        if failure is not None:
            _raise_worker_unavailable("worker request failed", failure)
        raise AssertionError("worker request unexpectedly completed without output")

    def upscale(self, data: WorkerInput, **parameters: object) -> WorkerOutput:
        return self._post(f"{self.upscale_url}/internal/upscale", data, parameters)

    def background(self, data: WorkerInput, **parameters: object) -> WorkerOutput:
        return self._post(f"{self.background_url}/internal/background-removal", data, parameters)

    def generation(self, request: dict[str, object]) -> WorkerOutput:
        try:
            response = self.client.post(
                f"{self.urls['generation']}/internal/generate", json=request
            )
            response.raise_for_status()
            return response.content
        except Exception as exc:
            _raise_worker_unavailable(
                "generation worker request failed", _sanitize_peer_failure(exc)
            )
        raise AssertionError("unreachable")

    def image_edit(self, data: WorkerInput, **parameters: object) -> WorkerOutput:
        return self._post(f"{self.urls['generation']}/internal/image-edit", data, parameters)


class PeerEvictor:
    """Call private peer unload controls while the caller already owns the global lane."""

    def __init__(
        self,
        peer_urls: tuple[str, ...],
        timeout_seconds: float = 30.0,
        client_factory: Callable[[float], httpx.Client] | None = None,
    ) -> None:
        self.peer_urls = tuple(url.rstrip("/") for url in peer_urls)
        self.timeout_seconds = timeout_seconds
        self.client_factory = client_factory or (
            lambda timeout: create_internal_http_client(httpx.Timeout(timeout))
        )

    def __call__(self) -> None:
        for peer_index, peer in enumerate(self.peer_urls):
            failure: SanitizedPeerFailure | None = None
            try:
                with self.client_factory(self.timeout_seconds) as client:
                    response = client.post(f"{peer}/internal/unload")
                    response.raise_for_status()
                    if response.json().get("unloaded") is not True:
                        raise ValueError("peer did not confirm unload")
            except Exception as exc:
                failure = _sanitize_peer_failure(exc)
            if failure is not None:
                _log_sanitized_peer_failure(
                    "peer model eviction failed: peer_index=%s", failure, peer_index
                )
                _raise_worker_unavailable("peer model eviction failed", failure)


class FakeWorkerClient:
    def __init__(self) -> None:
        self.model_invocations = 0
        self.model_loads = 0
        self.unload_calls = 0
        self.last_upscale: dict[str, object] = {}
        self.last_background: dict[str, object] = {}
        self._loaded: dict[str, str | None] = {
            "upscale": None,
            "background-removal": None,
            "generation": None,
        }

    def set_loaded(self, capability: str, model: str) -> None:
        self._loaded[capability] = model

    def health(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for capability, model in self._loaded.items():
            status: dict[str, object] = {
                "ready": True,
                "loaded": model is not None,
                "device": "cpu-test",
                "weightsAvailable": True,
            }
            if model is not None:
                status["loadedModel"] = model
            result[capability] = status
        return result

    def unload_all(self) -> dict[str, dict[str, object]]:
        self.unload_calls += 1
        for capability in self._loaded:
            self._loaded[capability] = None
        return {name: {"unloaded": True} for name in self._loaded}

    @staticmethod
    def _open(data: WorkerInput) -> Image.Image:
        stream = BytesIO(data) if isinstance(data, bytes) else data
        if not isinstance(data, bytes):
            stream.seek(0)
        with Image.open(stream) as image:
            return image.copy()

    def upscale(self, data: WorkerInput, **parameters: object) -> WorkerOutput:
        self.model_invocations += 1
        self.last_upscale = parameters
        opened = self._open(data)
        image = opened.convert("RGB")
        scale_value = parameters["outscale"]
        if not isinstance(scale_value, (int, float)):
            raise ValueError("fake worker outscale must be numeric")
        scale = float(scale_value)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        output = BytesIO()
        image.save(output, "PNG")
        return output.getvalue()

    def background(self, data: WorkerInput, **parameters: object) -> WorkerOutput:
        self.model_invocations += 1
        self.last_background = parameters
        image = self._open(data).convert("RGBA")
        output = BytesIO()
        image.save(output, "PNG")
        return output.getvalue()

    def generation(self, request: dict[str, object]) -> WorkerOutput:
        self.model_invocations += 1
        width, height = request.get("width"), request.get("height")
        if type(width) is not int or type(height) is not int:
            raise WorkerUnavailable("invalid generation request")
        output = BytesIO()
        Image.new("RGB", (width, height), (20, 30, 40)).save(output, "PNG")
        return output.getvalue()

    def image_edit(self, data: WorkerInput, **parameters: object) -> WorkerOutput:
        self.model_invocations += 1
        output = BytesIO()
        self._open(data).convert("RGB").save(output, "PNG")
        return output.getvalue()
