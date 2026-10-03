# Official FLUX.2 dev metadata fixture

Gzip files decompress byte-for-byte to config, index, processor and tokenizer files
from `diffusers/FLUX.2-dev-bnb-4bit` revision
`c30ad107542e63f222f864a8de510204394fb18a`. SHA-256 and byte sizes are recorded in
`src/image_api/model_manifests/flux-2-dev-bnb-4bit.json`.

Source: https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit/tree/c30ad107542e63f222f864a8de510204394fb18a

These are unchanged, pre-fetched upstream metadata, not inference weights.
Tests create sparse files of the exact published weight sizes to exercise physical
shard topology without downloading or loading models. Sparse fixture bytes are not
valid weights and do not claim weight checksum/inference verification. Heavy model,
CUDA and external-network boundaries are strict fakes. See NOTICE.md for separate
model licensing; fixture inclusion conveys no commercial-use authorization.
