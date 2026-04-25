"""엔드-투-엔드 파이프라인: 프레임 → YOLO → 발 픽셀 → 월드 좌표 → JSON.

사용자가 정의한 5단계 흐름을 한 스크립트로 엮은 것:
  1. 캘리브레이션 자동 체크 (없으면 RTSP에서 스냅샷 캡처 + calibrate 호출)
  2. ArUco id + 실측거리 + 스냅샷 → H 행렬 (기존 calibrate_homography.py 재사용)
  3. YOLO 탐지 → bbox 밑면 중점 (= 발 픽셀) 계산
  4. 발 픽셀을 H 행렬로 월드 좌표 변환
  5. cam1/cam2 관측을 하나의 JSON 페이로드로 직렬화

모드:
  # 단일 이미지 (테스트)
  python input/media/world_pipeline.py --cam cam1 --image calibration/test_cam1.jpg

  # 라이브 RTSP (실전)
  python input/media/world_pipeline.py --live
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# RTSP 전송을 TCP로 강제 (macOS + UDP 조합에서 cam 연결 실패/bus error 빈발).
# camera.py의 setdefault보다 먼저 설정해야 함.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32"
)

import cv2
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO  # noqa: E402

from calibrate_homography import run_calibration  # noqa: E402
from world_mapper import pixel_to_world  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
CALIBRATION_DIR = PROJECT_ROOT / "calibration"


# ── 작업자 식별용 ArUco ────────────────────────────────────────────────
# 작업자별 고유 ArUco 마커 (조끼 등에 부착). 검출되면 해당 worker_id 부여.
# workspace 코너 마커(22/24/27/38)와 ID 충돌 방지: 5, 10, 15 사용.
WORKER_ARUCO_MAP: dict[int, str] = {
    5: "W01",
    10: "W02",
    15: "W03",
}
_aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
_aruco_params = cv2.aruco.DetectorParameters()
# 작은/원거리 작업자 마커 감지 강화
_aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_aruco_params.adaptiveThreshWinSizeMin = 3
_aruco_params.adaptiveThreshWinSizeMax = 23
_aruco_params.adaptiveThreshWinSizeStep = 10
_aruco_params.minMarkerPerimeterRate = 0.01  # default 0.03 — 멀리 있는 작은 마커도 감지
_aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

# 환경변수 DEBUG_ARUCO=1 이면 매 프레임 감지된 모든 ID 콘솔 출력
_DEBUG_ARUCO = os.getenv("DEBUG_ARUCO", "0") == "1"


# ── 모델 로드 (모듈 로드 시 1회) ────────────────────────────────────────
pose_model = YOLO("yolo11n-pose.pt")

# 커스텀 모델 (현재: forklift + box_1 + box_2 탐지)
custom_model = None
custom_model_names: dict[int, str] = {}
_best_path = os.getenv("BEST_MODEL_PATH", "")
if _best_path and not os.path.isabs(_best_path):
    _best_path = str(PROJECT_ROOT / _best_path)
if _best_path and os.path.exists(_best_path):
    custom_model = YOLO(_best_path)
    custom_model_names = custom_model.names
    print(f"[init] 커스텀 모델 로드: {_best_path}")
    print(f"[init] 클래스: {custom_model_names}")
else:
    print("[init] 커스텀 모델 경로 없음 — person 감지만 수행")


# ── Step 3~4: YOLO + 발 픽셀 + 월드 변환 ────────────────────────────────
def extract_detections_with_world(frame, cam_id: str) -> list[dict]:
    """프레임 → YOLO → 발 픽셀 추출 → pixel_to_world → detection 리스트."""
    detections = []

    # (a) 사람 포즈 트래킹
    #  - 기본: YOLO-pose의 양 발목(15, 16) 중점을 "발 픽셀"로 사용
    #    → 카메라 각도에 무관하게 물리적 발 위치가 찍힘 (bbox는 각도따라 중심 이동함)
    #  - 발목 감지 실패 시 bbox 하단 중앙으로 폴백
    LEFT_ANKLE, RIGHT_ANKLE = 15, 16
    KPT_CONF_THRESHOLD = 0.3

    pose_results = pose_model.track(
        frame, conf=0.25, persist=True, verbose=False, classes=[0]
    )

    # (a-1) 작업자 식별용 ArUco 감지 (worker별 고유 ID)
    #   - 마커 중심점이 작업자 bbox 안(또는 마진 내)에 있으면 해당 worker_id 부여
    aruco_corners, aruco_ids, _ = _aruco_detector.detectMarkers(frame)
    worker_aruco_centers: dict[int, tuple[float, float]] = {}
    all_detected_ids: list[int] = []
    if aruco_ids is not None:
        for corner, mid in zip(aruco_corners, aruco_ids.flatten()):
            mid_int = int(mid)
            all_detected_ids.append(mid_int)
            if mid_int in WORKER_ARUCO_MAP:
                pts = corner[0]
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                worker_aruco_centers[mid_int] = (cx, cy)

    if _DEBUG_ARUCO:
        worker_hits = sorted(worker_aruco_centers.keys())
        print(
            f"[aruco/{cam_id}] all={sorted(all_detected_ids)}  "
            f"worker_ids={worker_hits}  (map={list(WORKER_ARUCO_MAP)})"
        )

    # 1) 모든 worker bbox + 발 픽셀 + 월드 좌표 수집 (worker_id 미정 상태로)
    worker_entries: list[dict] = []
    if pose_results[0].boxes is not None:
        boxes = pose_results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        ids_t = boxes.id
        ids_arr = (
            ids_t.cpu().numpy().astype(int).tolist()
            if ids_t is not None else [None] * len(xyxy)
        )

        kpts_xy = None
        kpts_conf = None
        if pose_results[0].keypoints is not None:
            kpts_xy = pose_results[0].keypoints.xy.cpu().numpy()       # (N, 17, 2)
            if pose_results[0].keypoints.conf is not None:
                kpts_conf = pose_results[0].keypoints.conf.cpu().numpy()  # (N, 17)

        for i, (box, tid) in enumerate(zip(xyxy, ids_arr)):
            x1, y1, x2, y2 = [float(v) for v in box]

            # 발목 키포인트 선호
            foot_x, foot_y = None, None
            foot_source = "bbox_bottom"
            if kpts_xy is not None and i < len(kpts_xy):
                la = kpts_xy[i, LEFT_ANKLE]
                ra = kpts_xy[i, RIGHT_ANKLE]
                la_ok = la[0] > 0 and la[1] > 0 and (
                    kpts_conf is None or kpts_conf[i, LEFT_ANKLE] >= KPT_CONF_THRESHOLD
                )
                ra_ok = ra[0] > 0 and ra[1] > 0 and (
                    kpts_conf is None or kpts_conf[i, RIGHT_ANKLE] >= KPT_CONF_THRESHOLD
                )
                if la_ok and ra_ok:
                    foot_x = (la[0] + ra[0]) / 2
                    foot_y = (la[1] + ra[1]) / 2
                    foot_source = "ankles_mid"
                elif la_ok:
                    foot_x, foot_y = float(la[0]), float(la[1])
                    foot_source = "left_ankle"
                elif ra_ok:
                    foot_x, foot_y = float(ra[0]), float(ra[1])
                    foot_source = "right_ankle"

            # 폴백: bbox 하단 중앙
            if foot_x is None:
                foot_x = (x1 + x2) / 2
                foot_y = y2

            wx, wy = pixel_to_world(foot_x, foot_y, cam_id)

            worker_entries.append({
                "type": "worker",
                "worker_id": None,           # 아래 매칭 패스에서 채움
                "track_id": tid,
                "bbox_px": [x1, y1, x2, y2],
                "foot_px": [float(foot_x), float(foot_y)],
                "foot_source": foot_source,
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })

    # 2) 마커 → 가장 가까운 bbox 매칭 (greedy, 1:1)
    #   기존 strict containment 의 실패 모드(마커가 bbox 가장자리 살짝 밖)를 해소.
    #   거리 임계값 = max(bbox_w, bbox_h) — bbox 한 변 길이 안의 마커는 그 워커로 인정.
    claimed_bboxes: set[int] = set()
    # 결정적 결과를 위해 마커 ID 오름차순으로 처리
    for mid_int in sorted(worker_aruco_centers.keys()):
        cx, cy = worker_aruco_centers[mid_int]
        best_idx = None
        best_dist = float("inf")
        for idx, entry in enumerate(worker_entries):
            if idx in claimed_bboxes:
                continue
            x1, y1, x2, y2 = entry["bbox_px"]
            bcx = (x1 + x2) / 2
            bcy = (y1 + y2) / 2
            max_dist = max(x2 - x1, y2 - y1)
            dist = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5
            if dist <= max_dist and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            worker_entries[best_idx]["worker_id"] = WORKER_ARUCO_MAP[mid_int]
            claimed_bboxes.add(best_idx)
            if _DEBUG_ARUCO:
                print(
                    f"[aruco/{cam_id}] marker {mid_int} → "
                    f"bbox#{best_idx} (dist={best_dist:.0f}px)"
                )
        elif _DEBUG_ARUCO:
            print(
                f"[aruco/{cam_id}] marker {mid_int} 매칭 실패 "
                f"(가장 가까운 bbox 가 거리 임계 초과 또는 bbox 0개)"
            )

    detections.extend(worker_entries)

    # (b) 커스텀 모델 감지 (forklift, box_1, box_2 등)
    #   - forklift : 지면 위 객체 → bbox 밑면 중심 (homography가 정확히 지면점으로 매핑)
    #   - box_1/box_2 : 크레인 인양물(공중) → bbox 밑면 중심 사용
    #     · 지면이 아닌 점은 homography에서 본질적 오차가 있지만,
    #       밑면이 bbox 중심보다 지면에 가깝기에 오차가 작음
    #     · 두 카메라 평균(realtime_camera.py)과 시간 평활화로 추가 보정
    if custom_model is not None:
        results = custom_model(frame, conf=0.5, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            cls_name = custom_model_names.get(cls_id, f"cls_{cls_id}")

            # 객체 타입별 기준점
            if cls_name in ("box_1", "box_2"):
                # 공중 인양물: 밑면 중심 (지면 projection 오차 최소화)
                ref_x = (x1 + x2) / 2
                ref_y = y2
                ref_source = "bbox_bottom_center_airborne"
            elif cls_name == "forklift":
                # 지면 객체: 밑면 중심 (지면점)
                ref_x = (x1 + x2) / 2
                ref_y = y2
                ref_source = "bbox_bottom_center"
            else:
                # 기타: bbox 중심 (default)
                ref_x = (x1 + x2) / 2
                ref_y = (y1 + y2) / 2
                ref_source = "bbox_center"

            wx, wy = pixel_to_world(ref_x, ref_y, cam_id)
            detections.append({
                "type": cls_name,
                "track_id": None,
                "bbox_px": [x1, y1, x2, y2],
                "foot_px": [ref_x, ref_y],
                "ref_source": ref_source,
                "world": {"x": round(wx, 3), "y": round(wy, 3)},
            })

    return detections


def draw_annotated(frame, detections: list[dict]):
    """bbox + 발 점 + 월드 좌표 라벨 오버레이."""
    out = frame.copy()
    colors = {
        "worker": (0, 255, 255),   # 노랑 (BGR)
        "forklift": (0, 0, 255),   # 빨강
        "box_1": (0, 200, 0),      # 초록
        "box_2": (255, 0, 200),    # 자주
    }
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_px"]]
        fx, fy = [int(v) for v in d["foot_px"]]
        c = colors.get(d["type"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.circle(out, (fx, fy), 10, (0, 0, 255), -1)
        cv2.circle(out, (fx, fy), 14, (255, 255, 255), 2)
        wx, wy = d["world"]["x"], d["world"]["y"]
        label = f'{d["type"]} ({wx:.2f}, {wy:.2f})m'
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return out


# ── Step 1~2: 캘리브레이션 자동 확보 ────────────────────────────────────
def _capture_rtsp_snapshot(cam_id: str, rtsp_url: str, interactive: bool = True) -> Path:
    """RTSP에서 1프레임 캡처하여 calibration/test_{cam_id}.jpg 저장.

    스레드 기반 VideoStream 대신 cv2.VideoCapture를 직접 사용.
    macOS + FFmpeg에서 스레드 VideoStream 재생성 시 Bus error가 자주 나서
    원샷 캡처는 단순 경로로 처리.

    interactive=False 이면 Enter 대기 없이 즉시 캡처 (서버 자동 기동용).
    """
    import gc
    snap = CALIBRATION_DIR / f"test_{cam_id}.jpg"
    print(f"[{cam_id}] 캘리브레이션 스냅샷 없음")
    if interactive:
        input(f"  → {cam_id} 시야에 ArUco 마커 4개를 모두 배치하고 Enter: ")
    else:
        print(f"  → 3초 뒤 자동 캡처 (ArUco 마커 4개가 시야에 있어야 함)")
        time.sleep(3)

    # TCP 강제 (UDP는 macOS + FFmpeg에서 bus error 빈발)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    for _ in range(40):  # 최대 ~4초 워밍업
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


def ensure_calibration(cam_id: str, rtsp_url: str | None = None) -> None:
    """캘리브레이션 JSON이 없으면 스냅샷 확보 → calibrate_homography.py 실행."""
    H_path = CALIBRATION_DIR / f"{cam_id}_homography.json"
    if H_path.exists():
        print(f"[{cam_id}] ✓ 캘리브레이션 존재 ({H_path.name})")
        return

    snap = CALIBRATION_DIR / f"test_{cam_id}.jpg"
    if not snap.exists():
        if rtsp_url is None:
            raise FileNotFoundError(
                f"{snap} 없음. 다음 중 하나로 해결:\n"
                f"  · --live 모드로 RTSP에서 자동 캡처\n"
                f"  · calibration/ 에 직접 이미지 저장 후 재실행"
            )
        _capture_rtsp_snapshot(cam_id, rtsp_url)

    print(f"[{cam_id}] 캘리브레이션 실행...")
    run_calibration(cam_id, snap)


# ── 모드 1: 단일 이미지 테스트 ──────────────────────────────────────────
def run_image(cam_id: str, image_path: Path) -> dict:
    ensure_calibration(cam_id)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"이미지 로드 실패: {image_path}")

    detections = extract_detections_with_world(frame, cam_id)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "source": image_path.name,
        cam_id: detections,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    # annotated 이미지 저장
    annotated = draw_annotated(frame, detections)
    out_path = CALIBRATION_DIR / f"{cam_id}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n✓ annotated: {out_path}")
    return payload


# ── 모드 2: 라이브 RTSP ─────────────────────────────────────────────────
def run_live(show: bool = True, interactive: bool = True) -> None:
    from camera import VideoStream

    rtsp_1 = os.getenv("CAMERA_RTSP_URL_1")
    rtsp_2 = os.getenv("CAMERA_RTSP_URL_2")
    if not rtsp_1 or not rtsp_2:
        raise RuntimeError(".env의 CAMERA_RTSP_URL_1, CAMERA_RTSP_URL_2 필요")

    # Phase 1: 두 카메라 스냅샷을 연속으로 먼저 확보
    # (캠1 캘리브레이션 후 캠2 캡처 시 FFmpeg 상태 오염 → bus error 방지)
    snap1 = CALIBRATION_DIR / "test_cam1.jpg"
    snap2 = CALIBRATION_DIR / "test_cam2.jpg"
    h1 = CALIBRATION_DIR / "cam1_homography.json"
    h2 = CALIBRATION_DIR / "cam2_homography.json"

    if not h1.exists() and not snap1.exists():
        _capture_rtsp_snapshot("cam1", rtsp_1, interactive=interactive)
    if not h2.exists() and not snap2.exists():
        _capture_rtsp_snapshot("cam2", rtsp_2, interactive=interactive)

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

    # 첫 프레임 도착까지 최대 25초 재시도 (카메라 이전 세션 만료 대기)
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

    ok1 = wait_for_frame(cam1, "cam1")
    ok2 = wait_for_frame(cam2, "cam2")

    if not ok2:
        print(
            "\n  ⚠ cam2 프레임 수신 실패. 가능 원인:\n"
            "    1) 카메라에 이전 RTSP 세션이 60초 이상 남아있음 — 시간을 두고 재시도\n"
            "    2) 카메라 전원/네트워크 이상 — 물리 재부팅 권장\n"
            "    3) stream2 URL 오류 — ffplay로 직접 테스트 (아래 안내 참조)\n"
        )
    print("[live] 파이프라인 진입 (ESC 종료)...")

    try:
        while True:
            ret1, f1 = cam1.read()
            ret2, f2 = cam2.read()

            d1 = extract_detections_with_world(f1, "cam1") if ret1 and f1 is not None else []
            d2 = extract_detections_with_world(f2, "cam2") if ret2 and f2 is not None else []

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--cam", help="cam1 | cam2 (이미지 모드)")
    parser.add_argument("--image", help="처리할 이미지 경로 (이미지 모드)")
    parser.add_argument("--live", action="store_true", help="RTSP 라이브 모드")
    parser.add_argument("--no-show", action="store_true", help="라이브 창 숨김")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Enter 대기 없이 자동 캡처 (서버 자동 기동용)")
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
