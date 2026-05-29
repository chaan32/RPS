"""이미지 USB 저장 / 서빙 서비스."""

from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..database.store import save_file
from ..database.store.service import LOCAL_FALLBACK_PATH, USB_BASE_PATH


async def upload(file: UploadFile) -> dict:
    """업로드된 이미지를 USB(또는 로컬 폴백) 에 저장하고 경로 반환."""
    contents = await file.read()
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{today}/{file.filename}"
    path = save_file(contents, key, content_type=file.content_type)
    return {"status": "success", "path": path, "filename": file.filename}


def _resolve_served_path(path: str) -> Path:
    """USB/local 저장소 prefix를 해석하고 저장소 밖 접근을 차단한다."""
    stores = {
        "usb": Path(USB_BASE_PATH).resolve(),
        "local": Path(LOCAL_FALLBACK_PATH).resolve(),
    }

    prefix, sep, rest = path.partition("/")
    if sep and prefix in stores:
        candidates = [(stores[prefix], rest)]
    else:
        candidates = [(stores["usb"], path), (stores["local"], path)]

    invalid_path = False
    for base, key in candidates:
        key_path = Path(key)
        target = key_path.resolve() if key_path.is_absolute() else (base / key_path).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            invalid_path = True
            continue
        if target.is_file():
            return target

    if invalid_path:
        raise HTTPException(status_code=400, detail="Invalid path")
    raise HTTPException(status_code=404, detail="Image not found")


def serve(path: str) -> FileResponse:
    """저장된 이미지를 HTTP 로 서빙한다. USB와 로컬 폴백을 모두 지원한다."""
    target = _resolve_served_path(path)
    try:
        target.stat()
    except OSError:
        raise HTTPException(status_code=404, detail="Image not found") from None
    return FileResponse(target)
