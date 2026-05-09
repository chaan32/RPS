"""ArUco 마커 검출 엔드포인트."""

from fastapi import APIRouter, File, UploadFile

from ..service import aruco_service

router = APIRouter()


@router.post("/aruco/identify")
async def aruco_identify(file: UploadFile = File(...)):
    """업로드 이미지의 ArUco 마커를 감지하고 라벨 그린 JPEG 반환.

    응답 헤더 X-Detected-IDs: 감지된 마커 ID 목록 (쉼표 구분).
    """
    return await aruco_service.identify(file)
