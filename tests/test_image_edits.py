from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from helpers import png
from PIL import Image

from image_api.app import create_app
from image_api.config import Settings
from image_api.workers import FakeWorkerClient, WorkerInput, WorkerOutput


def edit(client: TestClient, image: bytes | None = None, **fields: object):
    form = {
        "model": "longcat-image-edit",
        "prompt": "make the bee blue",
        "seed": "43",
        "negative_prompt": "",
    }
    form.update({name: str(value) for name, value in fields.items()})
    return client.post(
        "/v1/image-edits",
        data=form,
        files={"file": ("source.png", image or png("RGBA", (13, 7)), "image/png")},
    )


def test_image_edit_preserves_synchronous_png_contract(tmp_path) -> None:
    worker = FakeWorkerClient()
    client = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=worker))
    response = edit(client)
    assert response.status_code == 200
    with Image.open(BytesIO(response.content)) as image:
        assert image.mode == "RGB"
        assert image.size == (13, 7)
    assert worker.model_invocations == 1


def test_flux_2_klein_edit_preserves_source_and_exact_prompt(tmp_path) -> None:
    worker = FakeWorkerClient()
    client = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=worker))
    response = edit(client, model="flux-2-klein-4b", prompt="exact edit prompt")
    assert response.status_code == 200
    with Image.open(BytesIO(response.content)) as image:
        assert image.mode == "RGB"
        assert image.size == (13, 7)
    assert worker.model_invocations == 1


def test_non_flux_image_edit_passes_through_mismatched_worker_png(tmp_path) -> None:
    class MismatchedOutputWorker(FakeWorkerClient):
        def image_edit(self, data: WorkerInput, **parameters: object) -> WorkerOutput:
            self.model_invocations += 1
            return png("RGB", (13, 6))

    worker = MismatchedOutputWorker()
    client = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=worker))
    response = edit(client, model="longcat-image-edit")
    assert response.status_code == 200
    assert response.content == png("RGB", (13, 6))
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (13, 6)


def test_image_edit_validates_source_and_fields_before_dispatch(tmp_path) -> None:
    worker = FakeWorkerClient()
    client = TestClient(create_app(settings=Settings.for_tests(tmp_path), workers=worker))
    assert edit(client, b"not an image").status_code == 400
    assert edit(client, prompt="").status_code == 422
    assert edit(client, model="ideogram-4-nf4").status_code == 422
    assert worker.model_invocations == 0
