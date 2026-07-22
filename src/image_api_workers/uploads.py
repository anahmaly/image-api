from __future__ import annotations

from tempfile import SpooledTemporaryFile

from fastapi import HTTPException, UploadFile

UPLOAD_CHUNK_BYTES = 64 * 1024
UPLOAD_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024


async def read_bounded_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Spool an upload with a hard byte bound before making the adapter-required bytes object."""
    total = 0
    try:
        with SpooledTemporaryFile(max_size=UPLOAD_SPOOL_MEMORY_BYTES, mode="w+b") as spool:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                if len(chunk) > max_bytes - total:
                    raise HTTPException(413, "image upload exceeds configured limit")
                spool.write(chunk)
                total += len(chunk)
            spool.seek(0)
            return spool.read()
    finally:
        await file.close()
