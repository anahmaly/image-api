# image-api

Private-LAN gateway for isolated image workers: upscale, background removal, Ideogram generation, and LongCat image editing.

## Execution model

All public image requests are synchronous and ephemeral. The single gateway process owns one in-memory `SingleFlightCoordinator`; it admits at most one execution across all capabilities and returns `503` with `Retry-After: 1` when busy. The gateway forwards one bounded internal HTTP request to the selected worker and returns the existing PNG response.

No request, input, output, task, queue, or status is persisted. Restarting the gateway forgets in-flight work. A worker unavailable before inference returns retryable `503`; an interrupted or ambiguous request is never replayed by the service.

`GET /health` reports `ok` only when all required internal workers are ready, and `degraded` otherwise.

## Public API

- `GET /health` — worker readiness and coordinator capacity.
- `GET /v1/models` — supported models.
- `POST /v1/upscale` — synchronous multipart RGB PNG upscale.
- `POST /v1/background-removal` — synchronous multipart RGBA PNG background removal.
- `POST /v1/generations` — synchronous JSON RGB PNG generation.
- `POST /v1/image-edits` — synchronous multipart RGB PNG edit.
- `POST /v1/models/unload` — single-flight worker unload.

The gateway is the only Compose service that publishes a host port. Worker controls are internal.

## Development

```sh
uv sync --extra test --locked
.venv/bin/pytest -q tests/test_ephemeral_single_flight.py tests/test_background.py tests/test_cutover_contract.py
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src/image_api src/image_api_workers
```

Tests use deterministic fake workers and local images. They do not invoke model providers or GPU inference.
