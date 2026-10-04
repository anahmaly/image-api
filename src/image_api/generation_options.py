"""Request options shared by the gateway and generation worker."""


def validate_vae_tiling(model: object, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("vae_tiling must be a boolean")
    if value and model != "flux-2-dev-bnb-4bit":
        raise ValueError("vae_tiling is supported only for flux-2-dev-bnb-4bit")
    return value
