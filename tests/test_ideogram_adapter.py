from __future__ import annotations

import json
import os
import sys
import types
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from image_api_workers.ideogram import (
    IdeogramModel,
    IdeogramRuntimeUnavailable,
    _legacy_extra_special_tokens_compatibility,
    _qwen3_vl_tokenizer_config_compatibility,
)


class Pipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return [Image.new("RGB", (kwargs["width"], kwargs["height"]))]


def install_fake_ideogram(monkeypatch):
    presets = {
        "V4_DEFAULT_20": SimpleNamespace(
            num_steps=20, guidance_schedule=(7.0,) * 20, mu=0.0, std=1.75
        )
    }
    fake = SimpleNamespace(
        PRESETS=presets,
        aspect_ratio_from_size=lambda width, height: f"{width}:{height}",
    )
    monkeypatch.setitem(sys.modules, "ideogram4", fake)


def test_structured_caption_runs_offline_from_mounted_weights(tmp_path, monkeypatch) -> None:
    install_fake_ideogram(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    pipeline = Pipeline()
    status_path = tmp_path / "state" / "generation-model-status.json"
    model = IdeogramModel(
        weights,
        pipeline_factory=lambda: pipeline,
        cuda_available=lambda: True,
        status_path=status_path,
    )
    encoded = model(
        {
            "structured_caption": {
                "style": {"lighting": "soft", "palette": "warm"},
                "description": "a bee",
            },
            "width": 256,
            "height": 512,
            "seed": 7,
            "sampler_preset": "V4_DEFAULT_20",
        }
    )
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HOME"] == str(weights)
    assert json.loads(status_path.read_text()) == {"state": "loaded", "loaded": True}
    assert (
        pipeline.calls[0][0]
        == '{"style":{"lighting":"soft","palette":"warm"},"description":"a bee"}'
    )
    with Image.open(BytesIO(encoded)) as image:
        assert image.size == (256, 512)


def test_missing_weights_and_cuda_fail_honestly(tmp_path) -> None:
    missing = IdeogramModel(
        tmp_path / "missing", pipeline_factory=lambda: Pipeline(), cuda_available=lambda: True
    )
    try:
        missing._load()
        raise AssertionError("missing weights must fail")
    except IdeogramRuntimeUnavailable as exc:
        assert "unavailable" in str(exc)
    weights = tmp_path / "weights"
    weights.mkdir()
    no_cuda = IdeogramModel(
        weights, pipeline_factory=lambda: Pipeline(), cuda_available=lambda: False
    )
    try:
        no_cuda._load()
        raise AssertionError("missing CUDA must fail")
    except IdeogramRuntimeUnavailable as exc:
        assert "CUDA" in str(exc)


def test_legacy_list_extra_special_tokens_are_unnamed_and_mapping_behavior_is_unchanged(
    monkeypatch,
) -> None:
    class TokenizerBase:
        SPECIAL_TOKENS_ATTRIBUTES = ("bos_token",)

        def __init__(self) -> None:
            self._special_tokens_map = {}

        def _set_model_specific_special_tokens(self, special_tokens) -> None:
            self.SPECIAL_TOKENS_ATTRIBUTES = [
                *self.SPECIAL_TOKENS_ATTRIBUTES,
                *special_tokens.keys(),
            ]
            self._special_tokens_map.update(special_tokens)

    transformers = types.ModuleType("transformers")
    tokenization_utils_base = types.ModuleType("transformers.tokenization_utils_base")
    tokenization_utils_base.PreTrainedTokenizerBase = TokenizerBase
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules, "transformers.tokenization_utils_base", tokenization_utils_base
    )
    payload = [
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    tokenizer = TokenizerBase()

    with _legacy_extra_special_tokens_compatibility():
        tokenizer._set_model_specific_special_tokens(payload)
        tokenizer._set_model_specific_special_tokens({"image_token": "<image>"})

    assert tokenizer._extra_special_tokens == payload
    assert tokenizer._special_tokens_map == {"image_token": "<image>"}
    assert tokenizer.SPECIAL_TOKENS_ATTRIBUTES == ["bos_token", "image_token"]
    assert (
        TokenizerBase._set_model_specific_special_tokens.__name__
        == "_set_model_specific_special_tokens"
    )


def test_qwen3_vl_tokenizer_load_uses_sibling_text_encoder_config_only_for_tokenizer(
    monkeypatch,
) -> None:
    calls = []

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("config", path, kwargs))
            return "text-encoder-config"

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return "tokenizer"

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = AutoConfig
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    original_descriptor = AutoTokenizer.__dict__["from_pretrained"]
    with _qwen3_vl_tokenizer_config_compatibility():
        assert AutoTokenizer.from_pretrained("weights", subfolder="tokenizer") == "tokenizer"
        assert AutoTokenizer.from_pretrained("weights", subfolder="other") == "tokenizer"

    assert calls == [
        ("config", "weights", {"subfolder": "text_encoder"}),
        ("tokenizer", "weights", {"subfolder": "tokenizer", "config": "text-encoder-config"}),
        ("tokenizer", "weights", {"subfolder": "other"}),
    ]
    assert AutoTokenizer.__dict__["from_pretrained"] is original_descriptor


def test_plain_prompt_never_fakes_magic_prompt_success(tmp_path, monkeypatch) -> None:
    install_fake_ideogram(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.delenv("IMAGE_API_MAGIC_PROMPT_BACKEND", raising=False)
    monkeypatch.delenv("IMAGE_API_MAGIC_PROMPT_API_KEY", raising=False)
    model = IdeogramModel(weights, pipeline_factory=lambda: Pipeline(), cuda_available=lambda: True)
    try:
        model(
            {
                "prompt": "plain",
                "width": 256,
                "height": 256,
                "seed": 0,
                "sampler_preset": "V4_DEFAULT_20",
            }
        )
        raise AssertionError("unconfigured magic prompt must fail")
    except IdeogramRuntimeUnavailable as exc:
        assert "magic prompt" in str(exc)


def test_magic_prompt_provider_details_are_sanitized(tmp_path, monkeypatch) -> None:
    install_fake_ideogram(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setenv("IMAGE_API_MAGIC_PROMPT_BACKEND", "configured")
    monkeypatch.setenv("IMAGE_API_MAGIC_PROMPT_API_KEY", "not-a-real-key")

    class FailingExpander:
        def expand(self, *_args, **_kwargs):
            raise RuntimeError("private provider response body")

    model = IdeogramModel(
        weights,
        pipeline_factory=lambda: Pipeline(),
        magic_prompt_factory=lambda _backend: FailingExpander(),
        cuda_available=lambda: True,
    )
    try:
        model(
            {
                "prompt": "plain",
                "width": 256,
                "height": 256,
                "seed": 0,
                "sampler_preset": "V4_DEFAULT_20",
            }
        )
        raise AssertionError("provider failure must not succeed")
    except IdeogramRuntimeUnavailable as exc:
        assert str(exc) == "magic prompt expansion failed"
        assert exc.__cause__ is None
