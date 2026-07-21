from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from image_api_workers.ideogram import IdeogramModel, IdeogramRuntimeUnavailable


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
            "structured_caption": {"description": "a bee"},
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
    assert pipeline.calls[0][0] == '{"description":"a bee"}'
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
