# image-api

Private-LAN image processing gateway with process-isolated GPU capability workers. This is an atomic API cutover: there are no legacy public service, route, container, or environment aliases.

## Architecture

Only the `image-api` gateway publishes port `8000`. Upscaling, background removal, and generation execute in isolated worker environments. Every real worker holds the same `/state/gpu-lane.lock` OS file lock through inference and postprocessing, so a gateway timeout or disconnect cannot release GPU capacity while native work continues. Gateway reads of internal worker image responses are streamed into a configured-size capped buffer.

Generation admissions and state transitions are stored in SQLite with full synchronous durability. On restart, each running task is reconciled against only its canonical task-bound PNG: an exact RGB PNG with the requested dimensions and encoded-output bound is durably completed, while missing or invalid output and temporary files are cleaned before conservative terminal failure. Queued tasks remain claimable, and interrupted tasks are never resubmitted. Final PNG files are fsynced and atomically renamed before the durable success transition.

## Public API

- `GET /health` — bounded worker/capability/device/readiness metadata; does not load a model.
- `GET /v1/models` — supported models, presets, and dimension bounds; no secrets or host paths.
- `POST /v1/upscale` — multipart `file`; required `model`, `outscale`, and `tile`; returns PNG.
- `POST /v1/background-removal` — multipart `file`; required `model`; returns same-size PNG RGBA.
- `POST /v1/generations` — JSON body and required `Idempotency-Key`; persists and returns `202`.
- `GET /v1/generations/{taskId}` — queued/running/succeeded/failed status.
- `GET /v1/generations/{taskId}/image` — final PNG after success.

### Upscale

Supported model IDs:

- `RealESRGAN_x4plus`
- `RealESRGAN_x4plus_anime_6B`

`outscale` is `1–4`. `tile` is `0` or a multiple of 32 through 1024. ClipArtShop's required contract is:

```sh
curl -f -X POST \
  'http://HOST:8000/v1/upscale?model=RealESRGAN_x4plus&outscale=2&tile=512' \
  -F 'file=@input.png' -o output.png
```

### Background removal

Supported model IDs are `bria-rmbg-2.0` and `birefnet-hr-matting`. The gateway bounds alpha refinement and BiRefNet inference-size options before dispatch. Both models use local-mounted weights only.

```sh
curl -f -X POST \
  'http://HOST:8000/v1/background-removal?model=birefnet-hr-matting&birefnet_inference_size=2048' \
  -F 'file=@input.png' -o foreground.png
```

### Ideogram 4 generation

Dimensions must be multiples of 16 from 256 through 2048. Supported presets are `V4_QUALITY_48`, `V4_DEFAULT_20`, and `V4_TURBO_12`. Structured captions are accepted directly and require no hosted prompt service:

```sh
curl -f -X POST http://HOST:8000/v1/generations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: product-123-revision-4' \
  -d '{
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "sampler_preset": "V4_DEFAULT_20",
    "structured_caption": {"description": "A blue ceramic bee on a clean white surface"}
  }'
```

A replay with the same key and canonical request returns the original task. Reusing the key for a different request returns `409`. Plain prompts require `magic_prompt=true` and an explicitly configured `IMAGE_API_MAGIC_PROMPT_BACKEND`; missing configuration or expansion failure is never represented as success.

## Model mounts and licensing

No model weights are included or downloaded at request time. Production expects read-only operator mounts:

- `IMAGE_API_UPSCALE_WEIGHTS_HOST_PATH`: official Real-ESRGAN `.pth` files.
- `IMAGE_API_BRIA_WEIGHTS_HOST_PATH`: BRIA RMBG-2.0 model directory.
- `IMAGE_API_BIREFNET_WEIGHTS_HOST_PATH`: BiRefNet HR model directory.
- `IMAGE_API_IDEOGRAM_WEIGHTS_HOST_PATH`: a complete gated Ideogram 4 NF4 Hugging Face cache mount, including the repository `refs/main` and snapshot files used by the official pipeline.

The Ideogram worker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; NF4 fails honestly without the mount or CUDA. Tokens are neither command arguments nor required for production inference. Magic-prompt credentials, if enabled, are provided only through `IMAGE_API_MAGIC_PROMPT_API_KEY`.

Software and model terms are separate. See `NOTICE.md` and `licenses/`. Operators are responsible for every mounted model's applicable terms, including any separate commercial Ideogram agreement. This repository does not claim or distribute gated weights or private license artifacts.

## Configuration

All repository-owned configuration uses the `IMAGE_API_` prefix. Important bounds include:

- `IMAGE_API_MAX_REQUEST_BYTES` (default `21000000`), enforced on the entire HTTP body before expensive work.
- `IMAGE_API_MAX_UPLOAD_BYTES` (default `20000000`), enforced with chunked reads.
- `IMAGE_API_MAX_INPUT_PIXELS` (default `40000000`).
- `IMAGE_API_MAX_OUTPUT_PIXELS` (default `80000000`).
- `IMAGE_API_MAX_QUEUE_DEPTH` (default `100`).
- `IMAGE_API_WORKER_TIMEOUT_SECONDS` and `IMAGE_API_LANE_TIMEOUT_SECONDS`.

Bind defaults to `0.0.0.0:8000`; restrict access with the host firewall to trusted LAN devices.

## Compose

Validate production configuration:

```sh
docker compose config
```

Run the isolated GPU deployment after mounting licensed weights:

```sh
docker compose up -d --build
```

CPU-only deterministic test mode uses fake workers and downloads no model weights:

```sh
docker compose -f compose.yml -f compose.test.yml up --build
```

## Development

```sh
uv sync --extra test
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src/image_api src/image_api_workers
.venv/bin/pytest
docker compose config
docker compose -f compose.yml -f compose.test.yml config
```

Tests use real gateway, image validation, SQLite durability, atomic publication, and cross-process file locking with deterministic fake model boundaries. They do not download weights or call models/providers.

## Source provenance

The cutover was based on these verified source heads:

- Gateway repository base: `ad145f1003164d23d1b4bcee769b85b88417d8a9`.
- `anahmaly/rembg-api` main: `093a635c01102207e5af1a61e64180865a0a1220`.
- Validated BiRefNet HR implementation: `anahmaly/rembg-api#7` head `dd7b6fd434cff2077ce6e9a0cab46fe254f26f1f`.
- Official Ideogram 4 source: `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`.

The background worker depends directly on the validated PR head; the generation worker depends directly on the official Ideogram commit. Neither upstream branch is merged or modified by this repository.
