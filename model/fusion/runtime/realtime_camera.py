"""실시간 카메라 + YAMnet → Fusion 모델 통합 검증.

데이터 흐름:
  [Cam1 (RTSP)] ──► YOLO+ArUco+homography ──► world coords (worker, forklift)
  [Cam2 (RTSP)] ─┘                                   │
                                                      ▼
  [ESP32 마이크 → /ws/audio → 서버 YAMnet] ──► /audio/score HTTP 폴링 ──► audio_score
                                                      │
                                                      ▼
                                          RealtimeInference.push(...)
                                                      │
                                                      ▼
                                          risk_matrix (1, 2)
                                                      │
                                                      ▼
                                          BEV 시각화 + 콘솔 알림

실행:
  conda activate venv
  python model/fusion/realtime_camera.py            # ESP32 audio + 라이브 카메라
  python model/fusion/realtime_camera.py --no-audio # audio=0.05 고정 (오디오 무시)

ESC: 종료
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch  # noqa: F401  (사이드 이펙트 — CUDA/MPS 초기화)


# RTSP TCP 강제 (cv2가 import되기 전에 설정)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32",
)

# 프로젝트 경로 (이 파일: model/fusion/runtime/realtime_camera.py → 4단계 위가 PROJECT_ROOT)
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent.parent
# `python -m model.fusion.runtime.realtime_camera` 로 실행하면 sys.path 자동 처리되지만,
# 직접 실행이나 subprocess 환경 안전망으로 PROJECT_ROOT 만 명시 추가.
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Fusion 모듈
from ..inference import (
    load_dual_model,
    RealtimeInference,
    DEFAULT_THRESHOLD,
)
from ..risk_output import FusionPrediction, ThreatType
from ..data.scenario_generator import DZ_CENTER, DZ_RADIUS, RATE
from .publisher import publish_alert_via_server_sync
from .db_logger import log_pair_sync, log_pair_with_snapshot_sync

# 분리된 서브모듈
from . import audio_thread
from .kinematics import (
    WorkerKinematics,
    avg_speed,
    forklift_hazard_point,
    resolve_direction,
    DROPZONE_ALERT_RADIUS,
)
from .early_warning import (
    MotionHistory,
    evaluate_worker_forklift,
)
from .global_tracker import GlobalTrackManager
from .pair_builder import pick_positions
from .viz import draw_camera_overlay, render_bev

# 측정용 import
from server.utils.metrics import JsonLinesLogger
from server.utils.perf import add_camera_timings, add_duration_ms


# ── Headless 모드 감지 ─────────────────────────────────
# Docker 컨테이너처럼 디스플레이 서버가 없는 환경에선 cv2.imshow 가 Qt 로드 실패로
# 죽으므로, render_bev / draw_camera_overlay / imshow / waitKey 를 모두 skip.
# Dockerfile 의 QT_QPA_PLATFORM=offscreen 이 자동 트리거.
HEADLESS = (
    os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    or os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")
)


# ── 메인 루프 정책 상수 ────────────────────────────────
DZ_SMOOTHING_FRAMES = 5         # 인양물 BEV 좌표 시간 평활화 (median filter)
MAX_WORKERS = 3                 # 동시 추적 가능한 작업자 수


def _prefix_timing(
    prefix: str,
    timing: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    """Flatten one camera's DetectionPipeline timing into JSONL-safe keys."""
    return {
        f"{prefix}_{key}": value
        for key, value in timing.items()
        if key != "cam_id"
    }


def _count_detection_types(detections: list[dict]) -> dict[str, int]:
    """Count detection types for benchmark context."""
    counts: dict[str, int] = {}
    for det in detections:
        det_type = str(det.get("type", "unknown"))
        counts[det_type] = counts.get(det_type, 0) + 1
    return counts


# ── MQTT 발행 cooldown / fire-and-forget ───────────────
ALERT_COOLDOWN_SEC = 2.0
_last_publish_ts: dict[tuple[str, ThreatType], float] = {}


def _publish_in_background(
    pred: FusionPrediction,
    threshold: float,
    direction: str | None,
    frame_jpeg: bytes | None = None,
) -> None:
    """데몬 스레드에서 MQTT 발행 + DB 기록 — cv2 루프 블로킹 방지.

    순서: 아두이노 신호(MQTT) 먼저, 그 다음 DB 비동기 저장.
    direction 이 None 이면 작업자 정면 위험 등 진동 정책상 발행하지 않는
    상황이다. 이 경우에도 위험 판단 감사 로그는 남겨야 하므로 DB 저장은
    계속 수행한다.
    direction: 워커 body frame 기준 위협 방향에서 매핑된 펌웨어 명령
               ("back" | "left" | "right" | "all" | None). publisher 가 받아서
               /send-alert?direction=... 로 그대로 전달.
    frame_jpeg 있으면 스냅샷 업로드 엔드포인트 사용, 없으면 placeholder 경로.
    """
    # 1) /send-alert HTTP 호출 (server 가 MQTT 로 forklift/4/vibration 발행)
    if direction is None:
        print("  ↪︎  /send-alert skip: no haptic direction; DB log only")
    else:
        try:
            results = publish_alert_via_server_sync(
                pred, threshold=threshold, direction=direction,
            )
            for r in results:
                if r.get("status") == "success":
                    print(f"  📡 /send-alert → topic='{r.get('topic')}' "
                          f"payload='{r.get('message')}'  (dir={direction})")
                else:
                    print(f"  ⚠️  /send-alert fail: {r}")
        except Exception as e:
            print(f"  ⚠️  alert error: {e}")

    # 2) DB 저장 (incident_logs 비동기 기록)
    # snapshot 업로드(USB 미마운트, 권한 등)가 실패해도 incident_logs 행은
    # 반드시 남도록 placeholder 경로로 폴백한다.
    for p in pred.triggered(threshold):
        res: dict | None = None
        if frame_jpeg is not None:
            try:
                res = log_pair_with_snapshot_sync(p, frame_jpeg)
            except Exception as e:
                print(f"  ⚠️  DB log w/ snapshot exception: {e}")
                res = None
            if not res or res.get("status") != "ok":
                print(f"  ↪︎  snapshot 업로드 실패 → placeholder 경로로 재시도. detail={res}")
                try:
                    res = log_pair_sync(p)
                except Exception as e:
                    res = {"status": "fail", "error": f"fallback log_pair: {e}"}
        else:
            try:
                res = log_pair_sync(p)
            except Exception as e:
                res = {"status": "fail", "error": str(e)}

        if res and res.get("status") == "ok":
            snap = res.get("snapshot_path", "(placeholder)")
            print(f"  💾 DB log id={res.get('id')}  {p.threat_type.value} "
                  f"({p.level.value})  snapshot={snap}")
        else:
            print(f"  ⚠️  DB log fail: {res}")


def maybe_publish(
    pred: FusionPrediction,
    threshold: float,
    direction: str | None,
    frame=None,
) -> None:
    """cooldown 체크 후 위험한 PairRisk만 백그라운드 스레드로 기록/발행.

    direction: 위협 방향에서 매핑된 펌웨어 명령
               ("back" | "left" | "right" | "all" | None).
    frame: 알림 시점의 카메라 BGR 프레임 (cv2 ndarray). None이면 placeholder 경로 사용.
    """
    now = time.time()
    triggered_pairs = pred.triggered(threshold)
    fresh = [
        p for p in triggered_pairs
        if now - _last_publish_ts.get((p.worker_id, p.threat_type), 0.0) >= ALERT_COOLDOWN_SEC
    ]
    if not fresh:
        return
    for p in fresh:
        _last_publish_ts[(p.worker_id, p.threat_type)] = now

    # 메인 스레드에서 JPEG 인코딩 (cv2.imencode 는 빠름 — ~5ms)
    frame_jpeg: bytes | None = None
    if frame is not None:
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                frame_jpeg = buf.tobytes()
        except Exception as e:
            print(f"  ⚠️  JPEG encode error: {e}")

    threading.Thread(
        target=_publish_in_background,
        args=(pred, threshold, direction, frame_jpeg),
        daemon=True,
    ).start()


# ── 메인 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-audio", action="store_true", help="YAMnet 끄고 audio=0.05 고정")
    parser.add_argument("--no-cam2", action="store_true", help="cam1만 사용")
    parser.add_argument("--no-frames", action="store_true",
                        help="카메라 영상 창 숨김 (BEV만 표시)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="N초 후 자동 종료. 0이면 ESC/Ctrl-C 전까지 계속 실행.",
    )
    parser.add_argument(
        "--metrics-path",
        default=os.getenv("METRICS_PATH", "metrics/pipeline.jsonl"),
        help="프레임별 성능 JSONL 저장 경로.",
    )
    parser.add_argument(
        "--run-label",
        default=os.getenv("METRICS_RUN_LABEL", "default"),
        help="성능 로그에 남길 실행 라벨. 예: mac_only_s01",
    )
    parser.add_argument(
        "--extract-mode",
        choices=["serial", "camera_parallel", "model_parallel"],
        default=os.getenv("EXTRACT_MODE", "serial"),
        help=(
            "Detection extraction mode: serial, camera_parallel(2 threads), "
            "or model_parallel(4 threads)."
        ),
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=float(os.getenv("TARGET_RATE", RATE)),
        help="처리 목표 FPS. 0이면 sleep throttling 없이 가능한 만큼 처리.",
    )
    parser.add_argument(
        "--risk-engine",
        choices=["v1", "v2"],
        default=os.getenv("RISK_ENGINE", "v1"),
        help="위험 판단 엔진 선택. v1=기존 규칙/dual 모델, v2=GRU coordinate-window 모델.",
    )
    parser.add_argument(
        "--v2-checkpoint",
        type=Path,
        default=Path(os.getenv(
            "FUSION_V2_CHECKPOINT",
            str(PROJECT_ROOT / "model/fusion_v2/checkpoints_geometry_future/best.pt"),
        )),
        help="--risk-engine v2 일 때 사용할 Fusion V2 체크포인트.",
    )
    parser.add_argument(
        "--v2-window-size",
        type=int,
        default=int(os.getenv("FUSION_V2_WINDOW_SIZE", "24")),
        help="--risk-engine v2 일 때 사용할 좌표 window 크기.",
    )
    # 서버 subprocess 호환용 (world_pipeline에서 사용하는 인자, 여기선 무시)
    parser.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-prompt", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ── 위험 판단 엔진 로드 ──
    v2_factory = None
    if args.risk_engine == "v2":
        from model.fusion_v2.realtime import FusionV2RealtimeInference

        if not args.v2_checkpoint.exists():
            print(f"Fusion V2 체크포인트 없음: {args.v2_checkpoint}")
            sys.exit(1)
        v2_factory, v2_payload = FusionV2RealtimeInference.from_checkpoint(
            args.v2_checkpoint,
            device="cpu",
            window_size=args.v2_window_size,
        )
        model = None
        print(f"[fusion] V2 모델 로드 완료: {args.v2_checkpoint}")
        print(f"[fusion] V2 feature_dim={len(v2_payload['feature_columns'])} "
              f"window={args.v2_window_size}")
    else:
        # checkpoints/ 는 fusion/ root 에 있음 (이 파일은 fusion/runtime/ 에 있으므로 부모의 부모)
        ckpt_dir = _HERE.parent.parent / "checkpoints"
        if not ckpt_dir.exists():
            print(f"체크포인트 디렉터리 없음: {ckpt_dir}")
            sys.exit(1)
        model = load_dual_model(str(ckpt_dir), device="cpu")
        print(f"[fusion] dual 모델 로드 완료")

    # worker별 독립 RealtimeInference 인스턴스 관리.
    # 같은 모델 객체를 공유하므로 메모리 부담 없음.
    trackers: dict[str, object] = {}
    last_seen: dict[str, float] = {}
    kinematics: dict[str, WorkerKinematics] = {}
    EVICTION_SEC = 2.0   # N초 이상 미감지 시 tracker 제거

    # ── 카메라 + Detection Pipeline ──
    from input.media.camera import VideoStream
    from input.media.pipeline import DetectionRefiner, build_default_pipeline, ensure_calibration
    from input.media.pipeline.parallel import build_detection_executor

    detection_executor = build_detection_executor(args.extract_mode, build_default_pipeline)
    refiner = DetectionRefiner()
    global_tracker = GlobalTrackManager()

    rtsp1 = os.getenv("CAMERA_RTSP_URL_1")
    rtsp2 = os.getenv("CAMERA_RTSP_URL_2")
    if not rtsp1:
        print(".env CAMERA_RTSP_URL_1 필요")
        sys.exit(1)

    print("[cam] 캘리브레이션 확인 중...")
    interactive = not args.no_prompt
    ensure_calibration("cam1", rtsp_url=rtsp1, interactive=interactive)
    if not args.no_cam2:
        ensure_calibration("cam2", rtsp_url=rtsp2, interactive=interactive)

    print(f"[cam] cam1 연결: {rtsp1}")
    cam1 = VideoStream(rtsp1).start()

    cam2 = None
    if not args.no_cam2 and rtsp2:
        print(f"[cam] cam2 연결: {rtsp2}")
        cam2 = VideoStream(rtsp2).start()

    # 첫 프레임 대기
    def wait_frame(cam, name, max_wait=15.0):
        t0 = time.time()
        while time.time() - t0 < max_wait:
            ret, f = cam.read()
            if ret and f is not None:
                print(f"[cam] {name} 프레임 OK")
                return True
            time.sleep(0.3)
        print(f"[cam] {name} 프레임 timeout")
        return False

    wait_frame(cam1, "cam1")
    if cam2:
        wait_frame(cam2, "cam2")

    # ── 오디오 스레드 ──
    if not args.no_audio:
        threading.Thread(target=audio_thread.audio_worker, daemon=True).start()

    # ── 메인 루프 (5 FPS) ──
    print("\n[loop] 시작 (ESC 종료)\n")
    last_iter = 0.0
    iter_period = 1.0 / args.target_rate if args.target_rate > 0 else 0.0
    print_period = 0.2
    last_print = 0.0
    has_live_dz = False           # 인양물 검출되어 dropzone 갱신된 적 있는지

    # 인양물 BEV 좌표 시간 평활화 버퍼 (최근 N 프레임 median)
    dz_history: deque = deque(maxlen=DZ_SMOOTHING_FRAMES)

    # forklift 위치 history (정지 여부 판정용, 5Hz 기준 약 2초 분량)
    forklift_history: deque = deque(maxlen=10)
    forklift_hazard_motion = MotionHistory()
    worker_motion: dict[str, MotionHistory] = {}
    forklift_last_seen = 0.0

    # 현재 적용 중인 dropzone (live 모드 아니면 default)
    current_dz_center = np.array(DZ_CENTER, dtype=np.float32)
    current_dz_radius = float(DZ_RADIUS)

    # 메인 루프 돌아가기 직전
    # 성능 측정 로거 만들기 (파이프라인 단계별 latency 기록)
    metrics_logger = JsonLinesLogger(args.metrics_path)
    print(f"[metrics] path={args.metrics_path} run_label={args.run_label}")
    print(f"[metrics] extract_mode={args.extract_mode}")
    print(f"[metrics] target_rate={args.target_rate:g}")
    if args.duration > 0:
        print(f"[metrics] duration={args.duration:g}s")

    running = True
    loop_started_wall = time.time()
    frame_index = 0
    try:
        while running:
            now = time.time()
            if args.duration > 0 and now - loop_started_wall >= args.duration:
                print(f"[loop] duration reached: {args.duration:g}s")
                break
            if now - last_iter < iter_period:
                time.sleep(0.005)
                continue
            last_iter = now

            loop_wall_ts = time.time()
            t0 = time.perf_counter()
            ret1, f1 = cam1.read()
            ret2, f2 = (cam2.read() if cam2 else (False, None))
            t_read = time.perf_counter()

            extraction = detection_executor.extract_pair(
                f1,
                ret1 and f1 is not None,
                f2,
                ret2 and f2 is not None,
            )
            d1, d2 = extraction.cam1, extraction.cam2
            cam1_timing = extraction.cam1_timing
            cam2_timing = extraction.cam2_timing
            t_extract = time.perf_counter()

            detection_executor.cross_camera_propagate(
                {"cam1": d1, "cam2": d2},
                now_ts=now,
            )
            t_cross_camera = time.perf_counter()

            refined = refiner.refine({"cam1": d1, "cam2": d2})
            d1, d2 = refined["cam1"], refined["cam2"]
            t_refine = time.perf_counter()

            raw_workers_xy, raw_forklift_xy, raw_dropzone_xy = pick_positions(d1, d2)
            t_pick = time.perf_counter()

            workers_xy, forklift_xy, dropzone_xy = global_tracker.update(
                now,
                raw_workers_xy,
                raw_forklift_xy,
                raw_dropzone_xy,
            )
            t_global_track = time.perf_counter()

            # forklift 정지 여부 판정용 history 갱신.
            # forklift 가 1초 이상 안 보이면 history 비움 (옛 위치 + 새 위치
            # 사이의 가짜 이동거리로 잘못된 속도가 잡히는 걸 방지).
            if forklift_xy is not None:
                forklift_history.append(forklift_xy)
                forklift_last_seen = now
            elif forklift_history and (now - forklift_last_seen) > 1.0:
                forklift_history.clear()
            forklift_speed = avg_speed(forklift_history)
            forklift_hazard_xy = forklift_hazard_point(forklift_xy, forklift_history)
            if forklift_hazard_xy is not None:
                forklift_hazard_motion.update(now, forklift_hazard_xy)

            audio_score = audio_thread.get_score()

            # 인양물 검출되면 dz 좌표 갱신. 공중 객체는 homography 오차가 크므로
            # 최근 N 프레임 median 필터로 안정화.
            if dropzone_xy is not None:
                dz_history.append(dropzone_xy)
                xs = sorted(p[0] for p in dz_history)
                ys = sorted(p[1] for p in dz_history)
                mid = len(xs) // 2
                smoothed_dz = (xs[mid], ys[mid])
                current_dz_center = np.array(smoothed_dz, dtype=np.float32)
                has_live_dz = True
                # 모든 활성 tracker에 전파
                for tr in trackers.values():
                    tr.update_dropzone(center=smoothed_dz)

            crane_active = 0  # TODO: MQTT crane state
            t_motion_audio = time.perf_counter()

            # ── 멀티 워커: 각 worker_id 별 tracker/kinematics 갱신 + push ──
            for wid, wxy in workers_xy.items():
                if wid not in trackers:
                    if len(trackers) >= MAX_WORKERS:
                        continue   # 동시 추적 한도 초과
                    if args.risk_engine == "v2":
                        assert v2_factory is not None
                        trackers[wid] = v2_factory.create()
                    else:
                        trackers[wid] = RealtimeInference(model, device="cpu")
                    if has_live_dz:
                        trackers[wid].update_dropzone(
                            center=tuple(current_dz_center.tolist())
                        )
                    kinematics[wid] = WorkerKinematics()
                kinematics[wid].update(wxy)
                worker_motion.setdefault(wid, MotionHistory()).update(now, wxy)
                last_seen[wid] = now
                if args.risk_engine == "v2":
                    trackers[wid].push(
                        now_ts=now,
                        worker_xy=wxy,
                        forklift_xy=forklift_xy,
                        forklift_hazard_xy=forklift_hazard_xy,
                        dropzone_xy=tuple(current_dz_center.tolist()) if has_live_dz else None,
                        has_forklift=forklift_xy is not None,
                        has_dropzone=has_live_dz,
                    )
                else:
                    trackers[wid].push(forklift_hazard_xy, wxy, audio_score, crane_active)

            # 오래 안 보이는 worker tracker 제거
            stale = [wid for wid, ts in last_seen.items()
                     if now - ts > EVICTION_SEC]
            for wid in stale:
                trackers.pop(wid, None)
                kinematics.pop(wid, None)
                worker_motion.pop(wid, None)
                last_seen.pop(wid, None)

            t_tracker_push = time.perf_counter()

            # ── 워커별 risk 예측 ──
            risks_per_worker: dict[str, np.ndarray] = {}
            for wid, tr in trackers.items():
                if tr.ready():
                    risks_per_worker[wid] = tr.predict()  # (1, 2)
            t_fusion_predict = time.perf_counter()

            early_warnings = {}
            for wid, wxy in workers_xy.items():
                risk = risks_per_worker.get(wid)
                fusion_risk = float(risk[0, 0]) if risk is not None else None
                hist = worker_motion.setdefault(wid, MotionHistory())
                early_warnings[wid] = evaluate_worker_forklift(
                    worker_xy=wxy,
                    forklift_xy=forklift_hazard_xy,
                    worker_history=hist,
                    forklift_history=forklift_hazard_motion,
                    fusion_risk=fusion_risk,
                    fusion_threshold=args.threshold,
                )

            t_early_warning = time.perf_counter()

            # snapshot 용 프레임: cam1 우선, 없으면 cam2
            snapshot_frame = (
                f1 if (ret1 and f1 is not None)
                else f2 if (ret2 and f2 is not None)
                else None
            )

            # 위험한 워커가 1명이라도 있으면 MQTT/ DB 발행 (cooldown 은 maybe_publish 가 처리)
            # direction 은 워커 body frame 기준 위협 방향을 4방향에 투영해
            # 펌웨어 명령(back/left/right/all/None)으로 매핑한 값.
            #
            # 정책 보강:
            #   1) dropzone 은 fusion 출력과 무관하게 워커가 DROPZONE_ALERT_RADIUS
            #      안에 들면 dropzone risk 를 1.0 으로 강제 격상 (idle 인양물도 위험).
            #   2) forklift 가 정지 상태면 forklift trigger 를 무시 (resolve_direction).
            #   3) 위협 방향이 정면(front)이면 알림 X (resolve_direction → None).
            dz_xy_for_alert = tuple(current_dz_center.tolist())   # live 또는 default
            for wid, risk in risks_per_worker.items():
                wxy = workers_xy.get(wid)
                dz_force = False
                if wxy is not None:
                    d_wd = math.hypot(
                        wxy[0] - dz_xy_for_alert[0],
                        wxy[1] - dz_xy_for_alert[1],
                    )
                    if d_wd <= DROPZONE_ALERT_RADIUS:
                        # fusion 모델이 dropzone 을 0.0 으로 내도 강제로 1.0 격상.
                        # risk 는 ndarray 라 .copy() 로 원본 보호.
                        dz_force = True
                        risk = risk.copy()
                        risk[0, 1] = max(float(risk[0, 1]), 1.0)

                pred = FusionPrediction.from_model_output(risk, worker_ids=[wid])
                if not pred.has_alert(args.threshold):
                    continue

                direction = resolve_direction(
                    pred, args.threshold, kinematics[wid],
                    forklift_hazard_xy, dz_xy_for_alert,
                    forklift_speed=forklift_speed,
                    dz_force=dz_force,
                )
                maybe_publish(
                    pred, args.threshold, direction, frame=snapshot_frame,
                )
            t_publish_dispatch = time.perf_counter()
            # 콘솔 로그 (200ms마다)
            t_console_start = time.perf_counter()
            if risks_per_worker and now - last_print >= print_period:
                last_print = now
                ts_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                fork_str = (f"F=({forklift_xy[0]:+.2f},{forklift_xy[1]:+.2f})"
                            if forklift_xy else "F=(none)              ")
                print(f"t={ts_str}  audio={audio_score:.2f}  {fork_str}  "
                      f"workers={len(risks_per_worker)}")
                for wid, risk in risks_per_worker.items():
                    wxy = workers_xy.get(wid)
                    wstr = (f"({wxy[0]:+.2f},{wxy[1]:+.2f})"
                            if wxy else "(none)")
                    ew = early_warnings.get(wid)
                    ew_str = ""
                    if ew is not None and ew.level != "safe":
                        ttc = "--" if ew.ttc_s is None else f"{ew.ttc_s:.1f}s"
                        ca = "--" if ew.closest_distance_m is None else f"{ew.closest_distance_m:.2f}m"
                        label = ew.level.upper()
                        ew_str = f"  early={label} ttc={ttc} ca={ca}"
                    print(f"  - {wid} {wstr}  "
                          f"forklift_risk={risk[0,0]:.3f}  "
                          f"dropzone_risk={risk[0,1]:.3f}{ew_str}")
                    pred = FusionPrediction.from_model_output(risk, worker_ids=[wid])
                    for p in pred.triggered(args.threshold):
                        print(f"      🚨 {wid} ALERT → {p.threat_type.value} "
                              f"prob={p.prob:.3f} ({p.level.value})")
            t_console = time.perf_counter()

            # ── 시각화 (HEADLESS 면 통째로 skip) ──
            t_visual_start = time.perf_counter()
            if not HEADLESS:
                bev_dz_xy = tuple(current_dz_center.tolist()) if has_live_dz else None
                bev_dz_r = current_dz_radius if has_live_dz else None
                bev_headings = {wid: k.heading for wid, k in kinematics.items()}
                bev = render_bev(
                    workers_xy, forklift_xy, audio_score, risks_per_worker,
                    threshold=args.threshold,
                    dropzone_xy=bev_dz_xy, dropzone_radius=bev_dz_r,
                    forklift_hazard_xy=forklift_hazard_xy,
                    worker_headings=bev_headings,
                    early_warnings=early_warnings,
                )
                cv2.imshow("Fusion BEV", bev)

                if not args.no_frames:
                    if ret1 and f1 is not None:
                        overlay1 = draw_camera_overlay(
                            f1, d1, risks_per_worker, args.threshold
                        )
                        cv2.imshow("cam1 + risk", cv2.resize(overlay1, (900, 600)))
                    if ret2 and f2 is not None:
                        overlay2 = draw_camera_overlay(
                            f2, d2, risks_per_worker, args.threshold
                        )
                        cv2.imshow("cam2 + risk", cv2.resize(overlay2, (900, 600)))

                if cv2.waitKey(1) & 0xFF == 27:
                    running = False
            t_visual = time.perf_counter()

            worker_tracker_outliers = 0
            for wid in workers_xy:
                update = global_tracker.update_for(f"worker:{wid}")
                if update is not None:
                    worker_tracker_outliers += int(update.outlier)
            forklift_update = global_tracker.update_for("forklift")

            metrics = {
                "ts": loop_wall_ts,
                "run_label": args.run_label,
                "frame_index": frame_index,
                "risk_engine": args.risk_engine,
                "extract_mode": args.extract_mode,
                "target_rate": args.target_rate,
                "elapsed_s": round(loop_wall_ts - loop_started_wall, 6),
                "headless": int(HEADLESS),
                "duration_s": args.duration,
                "cam1_read_ok": int(ret1 and f1 is not None),
                "cam2_read_ok": int(ret2 and f2 is not None),
                "cam_read_ms": (t_read - t0) * 1000.0,
                "extract_pair_wall_ms": (t_extract - t_read) * 1000.0,
                "cam1_extract_wall_ms": float(cam1_timing.get("extract_total_ms", 0.0)),
                "cam2_extract_wall_ms": float(cam2_timing.get("extract_total_ms", 0.0)),
                "cross_camera_ms": (t_cross_camera - t_extract) * 1000.0,
                "refine_ms": (t_refine - t_cross_camera) * 1000.0,
                "pick_positions_ms": (t_pick - t_refine) * 1000.0,
                "global_track_ms": (t_global_track - t_pick) * 1000.0,
                "motion_audio_dz_ms": (t_motion_audio - t_global_track) * 1000.0,
                "tracker_push_ms": (t_tracker_push - t_motion_audio) * 1000.0,
                "fusion_forward_ms": (t_fusion_predict - t_tracker_push) * 1000.0,
                "early_warning_ms": (t_early_warning - t_fusion_predict) * 1000.0,
                "publish_dispatch_ms": (t_publish_dispatch - t_early_warning) * 1000.0,
                "console_ms": (t_console - t_console_start) * 1000.0,
                "visualize_ms": (t_visual - t_visual_start) * 1000.0,
                "loop_total_ms": (t_visual - t0) * 1000.0,
                # Backward-compatible coarse fields.
                "frame_read_extract_ms": (t_refine - t0) * 1000.0,
                "pick_push_ms": (t_tracker_push - t_refine) * 1000.0,
                "fusion_predict_ms": (t_early_warning - t_tracker_push) * 1000.0,
                "publish_ms": (t_publish_dispatch - t_early_warning) * 1000.0,
                "total_ms": (t_publish_dispatch - t0) * 1000.0,
                "n_workers": len(workers_xy),
                "n_predictions": len(risks_per_worker),
                "cam1_detection_counts": _count_detection_types(d1),
                "cam2_detection_counts": _count_detection_types(d2),
                "raw_workers": len(raw_workers_xy),
                "has_raw_forklift": int(raw_forklift_xy is not None),
                "has_raw_dropzone": int(raw_dropzone_xy is not None),
                "has_tracked_forklift": int(forklift_xy is not None),
                "has_forklift_hazard": int(forklift_hazard_xy is not None),
                "forklift_hazard_x": (
                    None if forklift_hazard_xy is None else round(float(forklift_hazard_xy[0]), 6)
                ),
                "forklift_hazard_y": (
                    None if forklift_hazard_xy is None else round(float(forklift_hazard_xy[1]), 6)
                ),
                "has_tracked_dropzone": int(dropzone_xy is not None),
                "worker_tracker_outliers": worker_tracker_outliers,
                "forklift_tracker_outlier": int(
                    forklift_update is not None and forklift_update.outlier
                ),
            }
            metrics.update(_prefix_timing("cam1", cam1_timing))
            metrics.update(_prefix_timing("cam2", cam2_timing))
            add_duration_ms(metrics, "perf.loop.total_ms", t0, t_visual)
            add_duration_ms(metrics, "perf.loop.total_without_visual_ms", t0, t_publish_dispatch)
            add_duration_ms(metrics, "perf.io.camera_read_ms", t0, t_read)
            add_duration_ms(metrics, "perf.pipeline.extract_pair_wall_ms", t_read, t_extract)
            add_duration_ms(metrics, "perf.pipeline.cross_camera_ms", t_extract, t_cross_camera)
            add_duration_ms(metrics, "perf.pipeline.refine_ms", t_cross_camera, t_refine)
            add_duration_ms(metrics, "perf.pipeline.pick_positions_ms", t_refine, t_pick)
            add_duration_ms(metrics, "perf.pipeline.global_track_ms", t_pick, t_global_track)
            add_duration_ms(metrics, "perf.pipeline.motion_audio_dz_ms", t_global_track, t_motion_audio)
            add_duration_ms(metrics, "perf.pipeline.tracker_push_ms", t_motion_audio, t_tracker_push)
            add_duration_ms(metrics, "perf.fusion.forward_ms", t_tracker_push, t_fusion_predict)
            add_duration_ms(metrics, "perf.fusion.early_warning_ms", t_fusion_predict, t_early_warning)
            add_duration_ms(metrics, "perf.output.publish_dispatch_ms", t_early_warning, t_publish_dispatch)
            add_duration_ms(metrics, "perf.ui.console_ms", t_console_start, t_console)
            add_duration_ms(metrics, "perf.ui.visualize_ms", t_visual_start, t_visual)
            add_camera_timings(metrics, "cam1", cam1_timing)
            add_camera_timings(metrics, "cam2", cam2_timing)
            metrics_logger.log(metrics)
            frame_index += 1

    finally:
        audio_thread.stop()
        detection_executor.shutdown()
        cam1.stop()
        if cam2:
            cam2.stop()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("\n[loop] 종료")


if __name__ == "__main__":
    main()
