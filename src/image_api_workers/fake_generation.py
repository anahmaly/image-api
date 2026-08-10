from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="image-api-test-generation-worker", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ready": True, "loaded": False, "device": "cpu-test", "weightsAvailable": True}


@app.post("/internal/unload")
def unload() -> dict[str, object]:
    return {"unloaded": True}


@app.post("/internal/generate")
def generate(request: dict[str, object]) -> Response:
    width, height = request.get("width"), request.get("height")
    if type(width) is not int or type(height) is not int:
        return Response(status_code=422)
    output = BytesIO()
    Image.new("RGB", (width, height), (20, 30, 40)).save(output, "PNG")
    return Response(output.getvalue(), media_type="image/png")


@app.post("/internal/image-edit")
async def edit(file: UploadFile = File()) -> Response:
    try:
        with Image.open(BytesIO(await file.read())) as source:
            output = BytesIO()
            source.convert("RGB").save(output, "PNG")
        return Response(output.getvalue(), media_type="image/png")
    finally:
        await file.close()
