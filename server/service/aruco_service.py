"""ArUco 마커 검출 + 라벨 그리기 서비스."""

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from fastapi.responses import Response

from input.media.tools.identify_markers import annotate_markers


async def identify(file: UploadFile) -> Response:
    """업로드 이미지에서 ArUco 마커 감지 → 라벨 그린 JPEG 반환.

    응답 헤더 X-Detected-IDs 에 감지된 마커 ID 쉼표 구분 문자열 포함.
    """
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    annotated, ids = annotate_markers(image)
    ok, buf = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="이미지 인코딩 실패")

    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={"X-Detected-IDs": ",".join(str(i) for i in ids)},
    )
