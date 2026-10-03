# image-api

Private-LAN gateway for isolated image workers: upscale, background removal, Ideogram generation, and LongCat image editing.

## Execution model

All public image requests are synchronous and ephemeral. The single gateway process owns one in-memory `SingleFlightCoordinator`; it admits at most one execution across all capabilities and returns `503` with `Retry-After: 1` when busy. The gateway forwards one bounded internal HTTP request to the selected worker and returns the existing PNG response.

No request, input, output, task, queue, or status is persisted. Restarting the gateway forgets in-flight work. A worker unavailable before inference returns retryable `503`; an interrupted or ambiguous request is never replayed by the service. A client disconnect does not abort a synchronous worker call: its slot remains owned until that call returns and the coordinator releases it in `finally`.

`GET /health` reports `ok` only when all required internal workers and required generation/edit models are ready, and `degraded` otherwise. The optional FLUX.2 dev selection does not affect aggregate readiness. The generation worker's container health is its responsive `/health` endpoint; its payload remains the authoritative per-model availability matrix. A selected unavailable generation/edit model is rejected before its internal worker dispatch, while another ready model remains usable.

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
Every valid model PNG response passes through with its produced bytes and dimensions unchanged.
The gateway retains non-dimension output validation: PNG encoding, required pixel mode, complete
decoding, and encoded transport byte ceilings. Callers own any optional dimension policy.

The gateway is the only Compose service that publishes a host port. Worker controls are internal.

## Request and snapshot bounds

The gateway separately enforces finite raw multipart-body ceilings before parsing and exact file-byte ceilings after parsing: 21 MB request / 20 MB file for normal routes, and 285 MB request / 280 MB file for processing routes. Compose exposes matching `IMAGE_API_*_REQUEST_BYTES` and `IMAGE_API_*_UPLOAD_BYTES` settings; the request allowance is bounded multipart framing, not file authority.

Ideogram and LongCat readiness accepts only the configured revision/ref marker or exact pinned snapshot directory, bounded parseable required JSON/config/tokenizer inputs, non-empty bounded merge files, and either direct weights or a bounded complete shard index with lexical absolute and `..` shard names rejected. Readiness validates mounted repository inputs only; it does not download or load models.

Production Compose mounts the existing model root once at `/models`. It resolves Ideogram at `/models/ideogram-4-nf4`, standard LongCat at `/models/longcat-image-edit`, and Turbo at `/models/longcat-image-edit-turbo`; `IMAGE_API_MODELS_HOST_PATH` defaults to `./models`.

## Optional FLUX.2 dev 32B NF4

`flux-2-dev-bnb-4bit` is a distinct, opt-in request selection, not an alias or
replacement for `flux-2-klein-4b`. Text generation accepts `model`, a direct
`prompt`, `seed`, `width` and `height` at `/v1/generations`; dimensions retain the
256–2048, multiple-of-16 input contract. Do not supply Ideogram caption expansion,
structured captions or sampler presets. `/v1/image-edits` accepts the same model,
`prompt`, `seed` and one uploaded image; negative prompts are unsupported. The
adapter converts the source to RGB and passes its exact dimensions to FLUX.2.
Upstream reference preprocessing (area limit and latent-grid alignment) remains
in the pinned pipeline; produced PNG dimensions pass through unchanged. Both paths
use 50 steps, guidance 4.0, one output and no caption upsampling.

The only supported artifact is
[`diffusers/FLUX.2-dev-bnb-4bit@c30ad107542e63f222f864a8de510204394fb18a`](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit/tree/c30ad107542e63f222f864a8de510204394fb18a),
with both official NF4 components. The checked-in
`src/image_api/model_manifests/flux-2-dev-bnb-4bit.json` records exact repository,
revision, paths, sizes and SHA-256 values. Default staging is
`/home/x/image-api/models/flux-2-dev-bnb-4bit`, mounted under the existing model root
at `/models/flux-2-dev-bnb-4bit`; `IMAGE_API_FLUX_2_DEV_WEIGHTS_PATH` selects the
container-local directory. Readiness requires `.image-api-revision` to contain the
exact revision, unchanged metadata (including quantization config and indexes),
four encoder shards, two transformer shards and one direct VAE safetensors file.
Missing, truncated, changed or unsupported layouts fail closed. Health checks
metadata hashes and weight sizes, not the full large weight checksums: the staging
owner must verify all manifest SHA-256 values before writing the revision marker.
Absent optional dev weights never prevent another ready model from running.

Inference is fully local and GPU-only: load/encode the quantized Mistral encoder on
CUDA, retain embeddings without an autograd graph, release the encoder, then load
the quantized transformer and CUDA VAE. Both VAE encode and decode run on CUDA.
No CPU offload, CPU inference fallback, automatic CPU/disk device map or remote
encoder is used for this selection. Every request releases both stages, including
failures, before a subsequent encoder can load. Existing models retain their prior
policies. The gateway lane and generation-child termination/reap boundary remain
the cross-model execution authority; same dev child reuse does not cache models.

The generation image retains Diffusers commit
`236e5dd9f38e21ae40c002539368b9be9a5e0fc8`, Transformers 5.5.0, Accelerate 1.11.0
and PyTorch 2.11.0, adding verified [`bitsandbytes==0.49.2`](https://pypi.org/project/bitsandbytes/0.49.2/).
Transformers 5.5.0 requires bitsandbytes >=0.46.1 / Accelerate >=1.1.0;
the pinned Diffusers 4-bit quantizer requires >=0.43.3 / >=0.26.0 respectively.
Exact-head CI builds the production Dockerfile and checks imports/stage APIs
without model construction or GPU allocation.

GPU-only peak-memory fit, host-RAM loading transients, throughput and real inference
remain untested. The upstream roughly 20GB CPU-offload example is not evidence of
GPU-only fit on a 24GB GPU. This feature does not download weights, accept model
terms, activate configuration, restart services or deploy. Search-first Nexus
inventory and verified staging are separate operator work. FLUX.2 dev model terms
are separate from this repository's MIT software license; an ungated download is
not a commercial-use permission. See `NOTICE.md`.

## Development

```sh
uv sync --extra test --locked
.venv/bin/pytest -q tests/test_ephemeral_single_flight.py tests/test_background.py tests/test_cutover_contract.py
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src/image_api src/image_api_workers
```

Tests use deterministic fake workers and local images. They do not invoke model providers or GPU inference.
