from __future__ import annotations

import struct
import sys
from io import BytesIO
from typing import cast

import pytest
from PIL import Image

from helpers import png
from image_api.app import create_app
from image_api.config import Settings
from image_api.store import TaskStore
from image_api.workers import FakeWorkerClient
from image_api_workers import background


def test_public_background_forwards_sam2_guidance_contract(tmp_path) -> None:
    worker = FakeWorkerClient()
    settings = Settings.for_tests(tmp_path)
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
        create_app(settings=settings, store=TaskStore(settings.database_path), workers=worker)
    )
    response = client.post(
        "/v1/background-removal?model=birefnet-hr-matting&sam2_guidance=true"
        "&sam2_model=sam2.1-hiera-large&sam2_mask_threshold=0.25"
        "&sam2_prompt_alpha_threshold=64&sam2_interior_erode=2"
        "&sam2_boundary_dilate=3&boundary_alpha_gamma=0.5",
        files={"file": ("input.png", png("RGB", (5, 3)), "image/png")},
    )
    assert response.status_code == 200
    assert worker.last_background["sam2_guidance"] is True
    assert worker.last_background["sam2_model"] == "sam2.1-hiera-large"
    assert worker.last_background["sam2_mask_threshold"] == 0.25
    assert worker.last_background["sam2_prompt_alpha_threshold"] == 64
    assert worker.last_background["sam2_interior_erode"] == 2
    assert worker.last_background["sam2_boundary_dilate"] == 3
    assert worker.last_background["boundary_alpha_gamma"] == 0.5


def test_guidance_fusion_is_pixelwise_monotonic_for_fragmented_sam_support() -> None:
    source = Image.new("RGB", (5, 1), (9, 8, 7))
    provisional = Image.frombytes("L", (5, 1), bytes([0, 64, 96, 255, 64]))
    sam_mask = Image.frombytes("L", (5, 1), bytes([0, 0, 0, 255, 0]))

    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=0,
        boundary_dilate=0,
        boundary_alpha_gamma=0.5,
        provisional_foreground_threshold=128,
    )

    adjusted = [0, 128, 156, 255, 128]
    alpha = list(fused.getchannel("A").getdata())
    assert all(final >= before for final, before in zip(alpha, adjusted, strict=True))
    assert alpha == adjusted
    assert fused.getpixel((0, 0))[:3] == (0, 0, 0)
    assert fused.getpixel((2, 0))[:3] == (9, 8, 7)


def test_guidance_preserves_enclosed_opening_wider_than_closing_kernel() -> None:
    source = Image.new("RGB", (15, 15), (9, 8, 7))
    provisional = Image.new("L", (15, 15), 0)
    sam_mask = Image.new("L", (15, 15), 0)
    for x in range(3, 12):
        sam_mask.putpixel((x, 3), 255)
        sam_mask.putpixel((x, 11), 255)
    for y in range(3, 12):
        sam_mask.putpixel((3, y), 255)
        sam_mask.putpixel((11, y), 255)

    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=0,
        boundary_dilate=1,
        boundary_alpha_gamma=1.0,
        provisional_foreground_threshold=128,
    )

    alpha = fused.getchannel("A")
    assert alpha.getpixel((3, 7)) == 255
    assert alpha.getpixel((5, 5)) == 0
    assert alpha.getpixel((0, 5)) == 0
    assert alpha.getpixel((14, 14)) == 0


def test_guidance_bounded_closing_repairs_pinhole_within_kernel() -> None:
    source = Image.new("RGB", (15, 15), (9, 8, 7))
    provisional = Image.new("L", (15, 15), 0)
    sam_mask = Image.new("L", (15, 15), 0)
    for x in range(3, 12):
        for y in range(3, 12):
            sam_mask.putpixel((x, y), 255)
    sam_mask.putpixel((7, 7), 0)

    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=0,
        boundary_dilate=1,
        boundary_alpha_gamma=1.0,
        provisional_foreground_threshold=128,
    )

    alpha = fused.getchannel("A")
    assert alpha.getpixel((7, 7)) == 255
    assert alpha.getpixel((0, 7)) == 0


def test_guidance_preserves_soft_provisional_alpha_inside_large_enclosed_opening() -> None:
    source = Image.new("RGB", (15, 15), (9, 8, 7))
    provisional = Image.new("L", (15, 15), 0)
    provisional.putpixel((5, 5), 64)
    sam_mask = Image.new("L", (15, 15), 0)
    for x in range(3, 12):
        sam_mask.putpixel((x, 3), 255)
        sam_mask.putpixel((x, 11), 255)
    for y in range(3, 12):
        sam_mask.putpixel((3, y), 255)
        sam_mask.putpixel((11, y), 255)

    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=0,
        boundary_dilate=1,
        boundary_alpha_gamma=0.5,
        provisional_foreground_threshold=128,
    )

    assert fused.getchannel("A").getpixel((5, 5)) == 128


def test_high_confidence_provisional_outside_eroded_support_remains_soft() -> None:
    source = Image.new("RGB", (7, 7), (9, 8, 7))
    provisional = Image.new("L", (7, 7), 0)
    provisional.putpixel((5, 5), 200)
    sam_mask = Image.new("L", (7, 7), 0)
    for x in range(1, 4):
        for y in range(1, 4):
            sam_mask.putpixel((x, y), 255)

    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=1,
        boundary_dilate=0,
        boundary_alpha_gamma=0.5,
        provisional_foreground_threshold=128,
    )

    assert fused.getchannel("A").getpixel((5, 5)) == 226


@pytest.mark.parametrize(
    ("probability", "size"),
    [
        (Image.new("F", (2, 2), 0.5), (3, 2)),
        (Image.frombytes("F", (3, 2), struct.pack("6f", float("nan"), *([0.5] * 5))), (3, 2)),
        (Image.new("F", (3, 2), 1.1), (3, 2)),
        (Image.new("F", (3, 2), 1.0), (3, 2)),
        (Image.new("F", (3, 2), 0.0), (3, 2)),
    ],
    ids=["wrong-size", "non-finite", "out-of-range", "all-one", "all-zero"],
)
def test_invalid_sam_masks_fail_explicitly(probability: Image.Image, size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="SAM"):
        background._binary_sam2_mask(probability, threshold=0.5, size=size)


def test_binary_sam2_mask_accepts_ordinary_probabilities_before_thresholding() -> None:
    probability = Image.frombytes("F", (2, 1), struct.pack("2f", 0.1, 0.9))

    mask = background._binary_sam2_mask(probability, threshold=0.5, size=(2, 1))

    assert list(mask.getdata()) == [0, 255]


def test_guidance_off_does_not_create_sam_adapter(monkeypatch) -> None:
    monkeypatch.setattr(background, "_sam2_adapter", None)
    created = False

    def fail_factory() -> object:
        nonlocal created
        created = True
        raise AssertionError("SAM must not load")

    monkeypatch.setattr(background, "_create_sam2_adapter", fail_factory)
    assert background._sam2_adapter is None
    assert created is False


def test_local_adapter_selects_best_full_resolution_logits_as_probabilities(monkeypatch) -> None:
    numpy = pytest.importorskip("numpy")
    adapter = object.__new__(background.LocalSam2Adapter)
    calls: dict[str, object] = {}

    class Predictor:
        def set_image(self, image) -> None:
            calls["image_size"] = image.shape[:2]

        def predict(self, **kwargs):
            calls.update(kwargs)
            full_resolution_logits = numpy.array(
                [
                    [[-2.0] * 5 for _ in range(3)],
                    [[0.0] * 5 for _ in range(3)],
                    [[2.0] * 5 for _ in range(3)],
                ],
                dtype=numpy.float32,
            )
            return numpy.zeros((3, 3, 5)), numpy.array([0.9, 0.9, 0.8]), full_resolution_logits

    adapter._predictor = Predictor()
    probability = adapter.predict_probability(Image.new("RGB", (5, 3)), (0, 0, 4, 2))

    assert calls["return_logits"] is True
    assert calls["image_size"] == (3, 5)
    assert probability.size == (5, 3)
    assert probability.mode == "F"
    assert probability.getpixel((0, 0)) == pytest.approx(0.5)


def test_guided_postprocessing_preserves_provisional_foreground_outside_sam_support(
    monkeypatch,
) -> None:
    from test_background_worker_adapter import _install_pr7_fakes

    calls: list[tuple[str, dict[str, object]]] = []
    _install_pr7_fakes(monkeypatch, calls)
    background._active_model = None

    class Adapter:
        def predict_probability(self, _source, _box):
            values = [0.0 if x == 0 else 1.0 for _y in range(7) for x in range(13)]
            return Image.frombytes("F", (13, 7), __import__("struct").pack("91f", *values))

        def release(self) -> None:
            pass

    monkeypatch.setattr(background, "_sam2_adapter", Adapter())
    processing = sys.modules["rembg_api.image_processing"]

    def postprocess(data: bytes, **_kwargs: object) -> bytes:
        with Image.open(BytesIO(data)) as image:
            processed = image.convert("RGBA")
            processed.putalpha(Image.new("L", processed.size, 200))
            output = BytesIO()
            processed.save(output, "PNG")
            return output.getvalue()

    processing.process_png_bytes = postprocess  # type: ignore[attr-defined]
    encoded = background._run_background(
        png("RGB", (13, 7)),
        model="birefnet-hr-matting",
        alpha_blur=1.5,
        alpha_erode=2,
        alpha_dilate=3,
        alpha_threshold=4,
        birefnet_inference_size=2048,
        birefnet_foreground_refinement=False,
        model_input_size=1024,
        sam2_guidance=True,
        sam2_interior_erode=0,
        sam2_boundary_dilate=0,
        boundary_alpha_gamma=0.6,
    )

    with Image.open(BytesIO(encoded)) as output:
        alpha = output.getchannel("A")
        adjusted_provisional_alpha = 226
        assert all(cast(int, final) >= adjusted_provisional_alpha for final in alpha.getdata())
        assert alpha.getpixel((0, 3)) == 255
        assert alpha.getpixel((1, 3)) == 255
        assert output.getpixel((0, 3))[:3] == (10, 20, 30)


def test_durable_runner_forwards_persisted_sam_contract(monkeypatch, tmp_path) -> None:
    settings = Settings.for_tests(tmp_path)
    store = TaskStore(settings.database_path)
    worker = FakeWorkerClient()
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
        create_app(settings=settings, store=store, workers=worker)
    )
    response = client.post(
        "/v1/background-removal-tasks?model=birefnet-hr-matting&sam2_guidance=true"
        "&sam2_model=sam2.1-hiera-large&sam2_mask_threshold=0.25"
        "&sam2_prompt_alpha_threshold=64&sam2_interior_erode=2"
        "&sam2_boundary_dilate=3&boundary_alpha_gamma=0.5",
        files={"file": ("input.png", png("RGB", (5, 3)), "image/png")},
        headers={"Idempotency-Key": "durable-sam-contract"},
    )
    assert response.status_code == 202
    captured: dict[str, object] = {}
    seen: dict[str, object] = {}
    monkeypatch.setenv("IMAGE_API_ENABLE_PROCESSING_RUNNER", "true")
    monkeypatch.setattr(background.Settings, "from_env", staticmethod(lambda: settings))
    monkeypatch.setattr(background, "PeerEvictor", lambda _peers: lambda: None)
    monkeypatch.setattr(
        background,
        "start_processing_runner",
        lambda _name, build_runner, **_kwargs: (
            captured.setdefault("runner", build_runner()) is not None
        ),
    )

    def run(data: bytes, **kwargs: object) -> bytes:
        seen.update(kwargs)
        return png("RGBA", (5, 3))

    monkeypatch.setattr(background, "_run_background", run)
    assert background.start_durable_runner()
    runner = captured["runner"]
    assert isinstance(runner, background.ProcessingRunner)
    assert runner.run_one()
    assert {
        key: seen[key]
        for key in (
            "sam2_guidance",
            "sam2_model",
            "sam2_mask_threshold",
            "sam2_prompt_alpha_threshold",
            "sam2_interior_erode",
            "sam2_boundary_dilate",
            "boundary_alpha_gamma",
        )
    } == {
        "sam2_guidance": True,
        "sam2_model": "sam2.1-hiera-large",
        "sam2_mask_threshold": 0.25,
        "sam2_prompt_alpha_threshold": 64,
        "sam2_interior_erode": 2,
        "sam2_boundary_dilate": 3,
        "boundary_alpha_gamma": 0.5,
    }
