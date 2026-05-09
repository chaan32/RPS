"""이미지 USB 저장 / 서빙 서비스."""

from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..database.store import save_file
from ..database.store.service import USB_BASE_PATH


async def upload(file: UploadFile) -> dict:
    """업로드된 이미지를 USB(또는 로컬 폴백) 에 저장하고 경로 반환."""
    contents = await file.read()
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{today}/{file.filename}"
    path = save_file(contents, key, content_type=file.content_type)
    return {"status": "success", "path": path, "filename": file.filename}


def serve(path: str) -> FileResponse:
    """USB 저장 이미지를 HTTP 로 서빙. 경로 탈출 방지."""
    base = Path(USB_BASE_PATH).resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(target)
