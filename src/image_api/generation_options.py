"""Request options shared by the gateway and generation worker."""

import re
from collections.abc import Mapping

MIN_INFERENCE_STEPS = 1
MAX_INFERENCE_STEPS = 50
FLUX_DEV_DEFAULT_STEPS = 25


def resolve_num_inference_steps(model: object, options: Mapping[str, object]) -> int | None:
    """Resolve only omission; explicitly supplied values must be strict integers."""
    if "num_inference_steps" not in options:
        return FLUX_DEV_DEFAULT_STEPS if model == "flux-2-dev-bnb-4bit" else None
    value = options["num_inference_steps"]
    if type(value) is not int or not MIN_INFERENCE_STEPS <= value <= MAX_INFERENCE_STEPS:
        raise ValueError("num_inference_steps must be an integer from 1 to 50")
    if model != "flux-2-dev-bnb-4bit":
        raise ValueError("num_inference_steps is supported only for flux-2-dev-bnb-4bit")
    return value


def parse_num_inference_steps(model: object, value: object) -> int | None:
    """Parse text transport fields without accepting booleans, fractions or blanks."""
    if value is None:
        return resolve_num_inference_steps(model, {})
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError("num_inference_steps must be an integer from 1 to 50")
    return resolve_num_inference_steps(model, {"num_inference_steps": int(value)})


def validate_vae_tiling(model: object, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("vae_tiling must be a boolean")
    if value and model != "flux-2-dev-bnb-4bit":
        raise ValueError("vae_tiling is supported only for flux-2-dev-bnb-4bit")
    return value
