"""Mock RTSP 검증용 합성 비디오 생성기.

ArUco 4코너 마커(22/24/27/38)와 가짜 "사람" 사각형이 들어간 mp4 두 개를 만든다.
이걸 mediamtx + ffmpeg 로 RTSP 송출하면 우리 코드가 진짜 카메라처럼 받아 처리.

검증 대상:
  ✅ cv2.VideoCapture 로 RTSP 디코딩
  ✅ ArUco 4코너 인식 → calibrate_homography 자동 동작
  ✅ DetectionPipeline 메인 루프 진입
  ❌ YOLO person/forklift 검출 (가짜 사각형은 못 잡음)
  ❌ Fusion 추론 (worker 검출 0건이라 입력 빔)

생성 파일:
  input/media/test/mock_rtsp/fake_cam1.mp4
  input/media/test/mock_rtsp/fake_cam2.mp4

실행:
  python -m input.media.test.make_mock_rtsp_video
"""

from pathlib import Path

import cv2
import numpy as np


# ── 비디오 파라미터 ────────────────────────────────────────────────────
W, H = 1280, 720
FPS = 30
DURATION_SEC = 30

OUT_DIR = Path(__file__).parent / "mock_rtsp"

# ── ArUco 마커 미리 렌더 ───────────────────────────────────────────────
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
MARKER_SIZE_PX = 200

# 작업공간 코너 마커 ID (calibration/world_markers.json 과 일치)
CORNER_IDS = [22, 24, 27, 38]

# 화면 픽셀 위치 — 4 코너에 배치.
# world_markers.json 의 좌표계와 매핑 (참고용):
#   22: (-2.22, 2.34) → 좌상
#   24: ( 0.00, 2.34) → 우상
#   27: (-2.22, 0.00) → 좌하
#   38: ( 0.00, 0.00) → 우하
MARGIN = 30
MARKER_POSITIONS = {
    22: (MARGIN, MARGIN),                                       # 좌상
    24: (W - MARKER_SIZE_PX - MARGIN, MARGIN),                  # 우상
    27: (MARGIN, H - MARKER_SIZE_PX - MARGIN),                  # 좌하
    38: (W - MARKER_SIZE_PX - MARGIN, H - MARKER_SIZE_PX - MARGIN),  # 우하
}


def _render_marker(mid: int) -> np.ndarray:
    img = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
    cv2.aruco.generateImageMarker(ARUCO_DICT, mid, MARKER_SIZE_PX, img, 1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def render_frame(t_sec: float, cam_id: str, markers: dict) -> np.ndarray:
    """t 초 시점의 합성 프레임 1장."""
    frame = np.full((H, W, 3), 200, dtype=np.uint8)  # 회색 배경

    # 4 코너 ArUco 마커
    for mid, (x, y) in MARKER_POSITIONS.items():
        frame[y:y + MARKER_SIZE_PX, x:x + MARKER_SIZE_PX] = markers[mid]

    # 가짜 "사람" — 좌우로 천천히 이동, 살짝 위아래 흔들림
    period = 10.0
    phase = 2 * np.pi * t_sec / period
    cx = int(W / 2 + (W / 4) * np.cos(phase))
    cy = int(H / 2 + 50 * np.sin(phase * 2))
    color = (60, 100, 200) if cam_id == "cam1" else (100, 60, 200)
    cv2.rectangle(frame, (cx - 50, cy - 100), (cx + 50, cy + 100), color, -1)
    cv2.putText(frame, "fake-person", (cx - 60, cy - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 메타 라벨
    cv2.putText(frame, f"{cam_id}  t={t_sec:5.1f}s",
                (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    return frame


def make_video(cam_id: str, out_path: Path, markers: dict) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 열기 실패: {out_path}")

    n_frames = FPS * DURATION_SEC
    for i in range(n_frames):
        t = i / FPS
        writer.write(render_frame(t, cam_id, markers))
    writer.release()
    print(f"  ✅ {out_path}  ({DURATION_SEC}초 / {n_frames} frames)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Mock RTSP 합성 비디오 생성 ({W}x{H} @ {FPS}fps × {DURATION_SEC}s) ===")
    print(f"  ArUco 코너 마커: {CORNER_IDS}")
    print(f"  출력 폴더: {OUT_DIR}\n")

    markers = {mid: _render_marker(mid) for mid in CORNER_IDS}

    make_video("cam1", OUT_DIR / "fake_cam1.mp4", markers)
    make_video("cam2", OUT_DIR / "fake_cam2.mp4", markers)

    print(f"\n다음 단계:")
    print(f"  1) mediamtx 실행:")
    print(f"     mediamtx")
    print(f"  2) ffmpeg 로 RTSP 송출 (각각 다른 터미널):")
    print(f"     ffmpeg -re -stream_loop -1 -i {OUT_DIR}/fake_cam1.mp4 \\")
    print(f"            -c copy -f rtsp rtsp://localhost:8554/cam1")
    print(f"     ffmpeg -re -stream_loop -1 -i {OUT_DIR}/fake_cam2.mp4 \\")
    print(f"            -c copy -f rtsp rtsp://localhost:8554/cam2")
    print(f"  3) .env 임시 변경:")
    print(f"     CAMERA_RTSP_URL_1=rtsp://localhost:8554/cam1")
    print(f"     CAMERA_RTSP_URL_2=rtsp://localhost:8554/cam2")
    print(f"  4) 검증:")
    print(f"     python -m input.media.test.verify_mock_rtsp")


if __name__ == "__main__":
    main()
