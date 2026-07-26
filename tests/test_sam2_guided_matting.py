from __future__ import annotations

import sys
from io import BytesIO

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


def test_guidance_fusion_preserves_hard_regions_and_gamma_unknown() -> None:
    source = Image.new("RGB", (5, 1), (9, 8, 7))
    provisional = Image.frombytes("L", (5, 1), bytes([0, 64, 128, 255, 255]))
    sam_mask = Image.frombytes("L", (5, 1), bytes([0, 255, 255, 255, 0]))

    fused = background.fuse_sam2_guidance(
        source,
        provisional,
        sam_mask,
        interior_erode=0,
        boundary_dilate=0,
        boundary_alpha_gamma=0.5,
    )

    assert list(fused.getchannel("A").getdata()) == [0, 255, 255, 255, 0]
    assert fused.getpixel((0, 0))[:3] == (0, 0, 0)
    assert fused.getpixel((2, 0))[:3] == (9, 8, 7)


@pytest.mark.parametrize(
    "mask",
    [
        Image.new("L", (3, 2), 0),
        Image.new("L", (3, 2), 255),
        Image.frombytes("F", (3, 2), b"\x00" * 24),
    ],
)
def test_invalid_sam_masks_fail_explicitly(mask: Image.Image) -> None:
    with pytest.raises(ValueError, match="SAM"):
        background.validate_sam2_mask(mask, (3, 2))


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


def test_guided_postprocessing_keeps_final_sam_hard_alpha_regions(monkeypatch) -> None:
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
        assert alpha.getpixel((0, 3)) == 0
        assert alpha.getpixel((1, 3)) == 255


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
