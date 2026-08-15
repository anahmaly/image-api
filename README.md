# image-api

Private-LAN gateway for isolated image workers: upscale, background removal, Ideogram generation, and LongCat image editing.

## Execution model

All public image requests are synchronous and ephemeral. The single gateway process owns one in-memory `SingleFlightCoordinator`; it admits at most one execution across all capabilities and returns `503` with `Retry-After: 1` when busy. The gateway forwards one bounded internal HTTP request to the selected worker and returns the existing PNG response.

No request, input, output, task, queue, or status is persisted. Restarting the gateway forgets in-flight work. A worker unavailable before inference returns retryable `503`; an interrupted or ambiguous request is never replayed by the service. A client disconnect does not abort a synchronous worker call: its slot remains owned until that call returns and the coordinator releases it in `finally`.

`GET /health` reports `ok` only when all required internal workers and every publicly selectable generation/edit model are ready, and `degraded` otherwise. The generation worker's container health is its responsive `/health` endpoint; its payload remains the authoritative per-model availability matrix. A selected unavailable generation/edit model is rejected before its internal worker dispatch, while another ready model remains usable.

## Public API

- `GET /health` — worker readiness and coordinator capacity.
- `GET /v1/models` — supported models.
- `POST /v1/upscale` — synchronous multipart RGB PNG upscale.
- `POST /v1/background-removal` — synchronous multipart RGBA PNG background removal.
- `POST /v1/generations` — synchronous JSON RGB PNG generation.
- `POST /v1/image-edits` — synchronous multipart RGB PNG edit.
- `POST /v1/models/unload` — single-flight worker unload.

Heavy models are globally single-resident: changing the selected model unloads the resident
model before replacement loading begins, while reusing the same loaded model does not cycle it.
LongCat edits remain normalized to source dimensions. Valid FLUX edit PNG bytes and dimensions
pass through unchanged; the gateway retains all other PNG validation.

The gateway is the only Compose service that publishes a host port. Worker controls are internal.

## Request and snapshot bounds

The gateway separately enforces finite raw multipart-body ceilings before parsing and exact file-byte ceilings after parsing: 21 MB request / 20 MB file for normal routes, and 285 MB request / 280 MB file for processing routes. Compose exposes matching `IMAGE_API_*_REQUEST_BYTES` and `IMAGE_API_*_UPLOAD_BYTES` settings; the request allowance is bounded multipart framing, not file authority.

Ideogram and LongCat readiness accepts only the configured revision/ref marker or exact pinned snapshot directory, bounded parseable required JSON/config/tokenizer inputs, non-empty bounded merge files, and either direct weights or a bounded complete shard index with lexical absolute and `..` shard names rejected. Readiness validates mounted repository inputs only; it does not download or load models.

Production Compose mounts the existing model root once at `/models`. It resolves Ideogram at `/models/ideogram-4-nf4`, standard LongCat at `/models/longcat-image-edit`, and Turbo at `/models/longcat-image-edit-turbo`; `IMAGE_API_MODELS_HOST_PATH` defaults to `./models`.

## Development

```sh
uv sync --extra test --locked
.venv/bin/pytest -q tests/test_ephemeral_single_flight.py tests/test_background.py tests/test_cutover_contract.py
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src/image_api src/image_api_workers
```

Tests use deterministic fake workers and local images. They do not invoke model providers or GPU inference.
