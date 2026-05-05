"""런타임 캘리브레이션 자동 보장.

- 기존 H 행렬 JSON 이 없으면 RTSP 에서 스냅샷을 1장 떠서 calibrate 실행.
- macOS + FFmpeg 의 bus error 를 피하려고 cv2.VideoCapture 단독 사용.
"""

import gc
import os
import time
from pathlib import Path

import cv2

from ..calibrate_homography import run_calibration


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"


def capture_rtsp_snapshot(
    cam_id: str,
    rtsp_url: str,
    interactive: bool = True,
) -> Path:
    """RTSP 에서 1프레임만 캡처해 calibration/test_{cam_id}.jpg 로 저장.

    스레드 기반 VideoStream 대신 cv2.VideoCapture 를 직접 사용.
    macOS + FFmpeg 에서 스레드 VideoStream 재생성 시 Bus error 가 자주 나서
    원샷 캡처는 단순 경로로 처리.

    interactive=False 이면 Enter 대기 없이 즉시 캡처 (서버 자동 기동용).
    """
    snap = CALIBRATION_DIR / f"test_{cam_id}.jpg"
    print(f"[{cam_id}] 캘리브레이션 스냅샷 없음")
    if interactive:
        input(f"  → {cam_id} 시야에 ArUco 마커 4개를 모두 배치하고 Enter: ")
    else:
        print(f"  → 3초 뒤 자동 캡처 (ArUco 마커 4개가 시야에 있어야 함)")
        time.sleep(3)

    # TCP 강제 (UDP 는 macOS + FFmpeg 에서 bus error 빈발)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    for _ in range(40):  # 최대 ~4 초 워밍업
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break
        time.sleep(0.1)
    cap.release()
    del cap
    gc.collect()
    time.sleep(0.5)  # FFmpeg 리소스 정리 시간

    if frame is None:
        raise RuntimeError(f"[{cam_id}] RTSP 프레임 캡처 실패: {rtsp_url}")

    CALIBRATION_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(snap), frame)
    print(f"  ✓ 캡처: {snap}")
    return snap


def ensure_calibration(
    cam_id: str,
    rtsp_url: str | None = None,
    interactive: bool = True,
) -> None:
    """캘리브레이션 JSON 이 없으면 스냅샷 확보 → calibrate_homography 실행.

    Args:
        cam_id:       "cam1" / "cam2"
        rtsp_url:     스냅샷이 없을 때 RTSP 에서 캡처용 (없으면 FileNotFoundError)
        interactive: True 면 Enter 대기, False 면 자동 캡처
    """
    H_path = CALIBRATION_DIR / f"{cam_id}_homography.json"
    if H_path.exists():
        print(f"[{cam_id}] ✓ 캘리브레이션 존재 ({H_path.name})")
        return

    snap = CALIBRATION_DIR / f"test_{cam_id}.jpg"
    if not snap.exists():
        if rtsp_url is None:
            raise FileNotFoundError(
                f"{snap} 없음. 다음 중 하나로 해결:\n"
                f"  · 라이브 모드로 RTSP 에서 자동 캡처\n"
                f"  · calibration/ 에 직접 이미지 저장 후 재실행"
            )
        capture_rtsp_snapshot(cam_id, rtsp_url, interactive=interactive)

    print(f"[{cam_id}] 캘리브레이션 실행...")
    run_calibration(cam_id, snap)
