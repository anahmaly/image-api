"""Build-time API/dependency contract. Imports only: no weights or GPU allocation."""

from importlib.metadata import version
from inspect import signature

from diffusers import (
    AutoencoderKLFlux2,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)
from diffusers.quantizers.bitsandbytes.bnb_quantizer import BnB4BitDiffusersQuantizer
from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

assert version("bitsandbytes") == "0.49.2"
assert version("transformers") == "5.5.0"
assert version("accelerate") == "1.11.0"
assert {"scheduler", "vae", "text_encoder", "tokenizer", "transformer"} <= set(
    signature(Flux2Pipeline.__init__).parameters
)
assert {"prompt", "device"} <= set(signature(Flux2Pipeline.encode_prompt).parameters)
assert {"image", "prompt_embeds", "width", "height", "caption_upsample_temperature"} <= set(
    signature(Flux2Pipeline.__call__).parameters
)
assert callable(AutoProcessor.from_pretrained)
assert callable(Mistral3ForConditionalGeneration.from_pretrained)
assert callable(Flux2Transformer2DModel.from_pretrained)
assert callable(AutoencoderKLFlux2.from_pretrained)
assert callable(BnB4BitDiffusersQuantizer.validate_environment)
assert callable(Bnb4BitHfQuantizer.validate_environment)
print("FLUX.2 dev pinned runtime import/API contract passed; no model loaded")
