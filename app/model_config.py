from typing import Dict, NamedTuple, Optional, Tuple


class ModelSpec(NamedTuple):
    canonical_name: str
    num_block: int
    network_scale: int
    model_path: str


STANDARD_MODEL = ModelSpec(
    canonical_name="RealESRGAN_x4plus",
    num_block=23,
    network_scale=4,
    model_path="/Real-ESRGAN/weights/RealESRGAN_x4plus.pth",
)

ANIME_6B_MODEL = ModelSpec(
    canonical_name="RealESRGAN_x4plus_anime_6B",
    num_block=6,
    network_scale=4,
    model_path="/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth",
)

SUPPORTED_CANONICAL_NAMES: Tuple[str, ...] = (
    STANDARD_MODEL.canonical_name,
    ANIME_6B_MODEL.canonical_name,
)

_MODELS_BY_SELECTION: Dict[str, ModelSpec] = {
    STANDARD_MODEL.canonical_name.casefold(): STANDARD_MODEL,
    ANIME_6B_MODEL.canonical_name.casefold(): ANIME_6B_MODEL,
    "Real-ESRGAN Anime 6B".casefold(): ANIME_6B_MODEL,
}


def resolve_model(value: Optional[str]) -> ModelSpec:
    """Resolve one immutable startup model selection."""
    selection = value.strip() if value is not None else ""
    if not selection:
        return STANDARD_MODEL

    try:
        return _MODELS_BY_SELECTION[selection.casefold()]
    except KeyError:
        supported = ", ".join(SUPPORTED_CANONICAL_NAMES)
        raise ValueError(
            "Unsupported REALESRGAN_MODEL {!r}. Supported models: {}".format(
                selection, supported
            )
        ) from None
