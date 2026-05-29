"""이미지 업로드 / 서빙 엔드포인트 (USB 또는 로컬 폴백)."""

from fastapi import APIRouter, File, UploadFile

from ..service import image_service

router = APIRouter()


@router.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """이미지를 USB(또는 로컬 폴백) 에 저장하고 저장 경로를 반환."""
    return await image_service.upload(file)


@router.get("/images/serve")
async def serve_usb_image(path: str):
    """저장 이미지를 HTTP 로 서빙. `<img src="/api/images/serve?path=...">` 용."""
    return image_service.serve(path)
