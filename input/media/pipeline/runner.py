"""파이프라인 실행 모드.

- run_image()  : 단일 이미지 1장으로 detection 테스트
- run_live()   : RTSP 듀얼 카메라 라이브 추론
- main()       : argparse CLI 진입점
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# RTSP 전송 TCP 강제 (cv2 import 전에 설정).
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32"
)

import cv2
from dotenv import load_dotenv

from .calibration_runtime import (
    CALIBRATION_DIR,
    PROJECT_ROOT,
    capture_rtsp_snapshot,
    ensure_calibration,
)
from ..calibrate_homography import run_calibration
from .engine import DetectionPipeline
from .visualization import draw_annotated


load_dotenv(PROJECT_ROOT / ".env")


def build_default_pipeline() -> DetectionPipeline:
    """env 기반으로 DetectionPipeline 인스턴스 생성."""
    return DetectionPipeline(
        pose_model_path="yolo11n-pose.pt",
        custom_model_path=os.getenv("BEST_MODEL_PATH", "") or None,
        debug_aruco=os.getenv("DEBUG_ARUCO", "0") == "1",
    )


# ── 모드 1: 단일 이미지 테스트 ──────────────────────────────────────────
def run_image(cam_id: str, image_path: Path) -> dict:
    ensure_calibration(cam_id)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    pipeline = build_default_pipeline()
    detections = pipeline.extract(frame, cam_id)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "source": image_path.name,
        cam_id: detections,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    annotated = draw_annotated(frame, detections)
    out_path = CALIBRATION_DIR / f"{cam_id}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n✓ annotated: {out_path}")
    return payload


# ── 모드 2: 라이브 RTSP ─────────────────────────────────────────────────
def run_live(show: bool = True, interactive: bool = True) -> None:
    from ..camera import VideoStream

    rtsp_1 = os.getenv("CAMERA_RTSP_URL_1")
    rtsp_2 = os.getenv("CAMERA_RTSP_URL_2")
    if not rtsp_1 or not rtsp_2:
        raise RuntimeError(".env 의 CAMERA_RTSP_URL_1, CAMERA_RTSP_URL_2 필요")

    # Phase 1: 두 카메라 스냅샷을 연속 확보
    # (캠1 캘리브레이션 후 캠2 캡처 시 FFmpeg 상태 오염 → bus error 방지)
    snap1 = CALIBRATION_DIR / "test_cam1.jpg"
    snap2 = CALIBRATION_DIR / "test_cam2.jpg"
    h1 = CALIBRATION_DIR / "cam1_homography.json"
    h2 = CALIBRATION_DIR / "cam2_homography.json"

    if not h1.exists() and not snap1.exists():
        capture_rtsp_snapshot("cam1", rtsp_1, interactive=interactive)
    if not h2.exists() and not snap2.exists():
        capture_rtsp_snapshot("cam2", rtsp_2, interactive=interactive)

    # Phase 2: 이미지 파일만으로 캘리브레이션 (RTSP 미사용)
    if not h1.exists():
        print(f"[cam1] 캘리브레이션 실행...")
        run_calibration("cam1", snap1)
    else:
        print(f"[cam1] ✓ 캘리브레이션 존재 ({h1.name})")

    if not h2.exists():
        print(f"[cam2] 캘리브레이션 실행...")
        run_calibration("cam2", snap2)
    else:
        print(f"[cam2] ✓ 캘리브레이션 존재 ({h2.name})")

    # Phase 3: 라이브 스트림 시작
    print("\n[live] 카메라 스트림 시작 중...")
    cam1 = VideoStream(rtsp_1).start()
    cam2 = VideoStream(rtsp_2).start()

    def wait_for_frame(cam, name: str, max_wait: float = 25.0) -> bool:
        start = time.time()
        while time.time() - start < max_wait:
            ret, frame = cam.read()
            if ret and frame is not None:
                print(f"  [{name}] ✓ 프레임 수신 ({int(time.time() - start)}s)")
                return True
            time.sleep(0.5)
        print(f"  [{name}] ✗ {int(max_wait)}s 내 프레임 없음")
        return False

    wait_for_frame(cam1, "cam1")
    ok2 = wait_for_frame(cam2, "cam2")
    if not ok2:
        print(
            "\n  ⚠ cam2 프레임 수신 실패. 가능 원인:\n"
            "    1) 카메라에 이전 RTSP 세션이 60초 이상 남아있음 — 시간을 두고 재시도\n"
            "    2) 카메라 전원/네트워크 이상 — 물리 재부팅 권장\n"
            "    3) stream2 URL 오류 — ffplay 로 직접 테스트\n"
        )
    print("[live] 파이프라인 진입 (ESC 종료)...")

    # Phase 4: pipeline 인스턴스 생성 후 메인 루프
    pipeline = build_default_pipeline()
    try:
        while True:
            ret1, f1 = cam1.read()
            ret2, f2 = cam2.read()

            d1 = pipeline.extract(f1, "cam1") if ret1 and f1 is not None else []
            d2 = pipeline.extract(f2, "cam2") if ret2 and f2 is not None else []

            pipeline.cross_camera_propagate({"cam1": d1, "cam2": d2})

            payload = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "cam1": d1,
                "cam2": d2,
            }
            print(json.dumps(payload, ensure_ascii=False))

            if show:
                if ret1 and f1 is not None:
                    cv2.imshow("cam1", cv2.resize(draw_annotated(f1, d1), (800, 600)))
                if ret2 and f2 is not None:
                    cv2.imshow("cam2", cv2.resize(draw_annotated(f2, d2), (800, 600)))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cam1.stop()
        cam2.stop()
        cv2.destroyAllWindows()


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam", help="cam1 | cam2 (이미지 모드)")
    parser.add_argument("--image", help="처리할 이미지 경로 (이미지 모드)")
    parser.add_argument("--live", action="store_true", help="RTSP 라이브 모드")
    parser.add_argument("--no-show", action="store_true", help="라이브 창 숨김")
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="Enter 대기 없이 자동 캡처 (서버 자동 기동용)",
    )
    args = parser.parse_args()

    if args.live:
        run_live(show=not args.no_show, interactive=not args.no_prompt)
    elif args.cam and args.image:
        run_image(args.cam, Path(args.image))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
