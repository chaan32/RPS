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
    resolve_direction,
    DROPZONE_ALERT_RADIUS,
)
from .pair_builder import pick_positions
from .viz import draw_camera_overlay, render_bev


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


# ── MQTT 발행 cooldown / fire-and-forget ───────────────
ALERT_COOLDOWN_SEC = 2.0
_last_publish_ts: dict[ThreatType, float] = {}


def _publish_in_background(
    pred: FusionPrediction,
    threshold: float,
    direction: str,
    frame_jpeg: bytes | None = None,
) -> None:
    """데몬 스레드에서 MQTT 발행 + DB 기록 — cv2 루프 블로킹 방지.

    순서: 아두이노 신호(MQTT) 먼저, 그 다음 DB 비동기 저장.
    direction: 워커 body frame 기준 위협 방향에서 매핑된 펌웨어 명령
               ("back" | "left" | "right" | "all"). publisher 가 받아서
               /send-alert?direction=... 로 그대로 전달.
    frame_jpeg 있으면 스냅샷 업로드 엔드포인트 사용, 없으면 placeholder 경로.
    """
    # 1) /send-alert HTTP 호출 (server 가 MQTT 로 forklift/4/vibration 발행)
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
    direction: str,
    frame=None,
) -> None:
    """cooldown 체크 후 위험한 PairRisk만 백그라운드 스레드로 발행.

    direction: 위협 방향에서 매핑된 펌웨어 명령
               ("back" | "left" | "right" | "all").
    frame: 알림 시점의 카메라 BGR 프레임 (cv2 ndarray). None이면 placeholder 경로 사용.
    """
    now = time.time()
    triggered_types = {p.threat_type for p in pred.triggered(threshold)}
    fresh = [
        t for t in triggered_types
        if now - _last_publish_ts.get(t, 0.0) >= ALERT_COOLDOWN_SEC
    ]
    if not fresh:
        return
    for t in fresh:
        _last_publish_ts[t] = now

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
    # 서버 subprocess 호환용 (world_pipeline에서 사용하는 인자, 여기선 무시)
    parser.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-prompt", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ── 모델 로드 (페어별 best dual) ──
    # checkpoints/ 는 fusion/ root 에 있음 (이 파일은 fusion/runtime/ 에 있으므로 부모의 부모)
    ckpt_dir = _HERE.parent.parent / "checkpoints"
    if not ckpt_dir.exists():
        print(f"체크포인트 디렉터리 없음: {ckpt_dir}")
        sys.exit(1)
    model = load_dual_model(str(ckpt_dir), device="cpu")
    print(f"[fusion] dual 모델 로드 완료")

    # worker별 독립 RealtimeInference 인스턴스 관리.
    # 같은 모델 객체를 공유하므로 메모리 부담 없음.
    trackers: dict[str, RealtimeInference] = {}
    last_seen: dict[str, float] = {}
    kinematics: dict[str, WorkerKinematics] = {}
    EVICTION_SEC = 2.0   # N초 이상 미감지 시 tracker 제거

    # ── 카메라 + Detection Pipeline ──
    from input.media.camera import VideoStream
    from input.media.pipeline import build_default_pipeline, ensure_calibration

    pipeline = build_default_pipeline()

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
    iter_period = 1.0 / RATE   # 0.2s
    print_period = 0.2
    last_print = 0.0
    has_live_dz = False           # 인양물 검출되어 dropzone 갱신된 적 있는지

    # 인양물 BEV 좌표 시간 평활화 버퍼 (최근 N 프레임 median)
    dz_history: deque = deque(maxlen=DZ_SMOOTHING_FRAMES)

    # forklift 위치 history (정지 여부 판정용, 5Hz 기준 약 2초 분량)
    forklift_history: deque = deque(maxlen=10)
    forklift_last_seen = 0.0

    # 현재 적용 중인 dropzone (live 모드 아니면 default)
    current_dz_center = np.array(DZ_CENTER, dtype=np.float32)
    current_dz_radius = float(DZ_RADIUS)

    running = True
    try:
        while running:
            now = time.time()
            if now - last_iter < iter_period:
                time.sleep(0.005)
                continue
            last_iter = now

            ret1, f1 = cam1.read()
            ret2, f2 = (cam2.read() if cam2 else (False, None))
            d1 = pipeline.extract(f1, "cam1") if ret1 and f1 is not None else []
            d2 = pipeline.extract(f2, "cam2") if ret2 and f2 is not None else []

            workers_xy, forklift_xy, dropzone_xy = pick_positions(d1, d2)

            # forklift 정지 여부 판정용 history 갱신.
            # forklift 가 1초 이상 안 보이면 history 비움 (옛 위치 + 새 위치
            # 사이의 가짜 이동거리로 잘못된 속도가 잡히는 걸 방지).
            if forklift_xy is not None:
                forklift_history.append(forklift_xy)
                forklift_last_seen = now
            elif forklift_history and (now - forklift_last_seen) > 1.0:
                forklift_history.clear()
            forklift_speed = avg_speed(forklift_history)

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

            # ── 멀티 워커: 각 worker_id 별 tracker/kinematics 갱신 + push ──
            for wid, wxy in workers_xy.items():
                if wid not in trackers:
                    if len(trackers) >= MAX_WORKERS:
                        continue   # 동시 추적 한도 초과
                    trackers[wid] = RealtimeInference(model, device="cpu")
                    if has_live_dz:
                        trackers[wid].update_dropzone(
                            center=tuple(current_dz_center.tolist())
                        )
                    kinematics[wid] = WorkerKinematics()
                kinematics[wid].update(wxy)
                last_seen[wid] = now
                trackers[wid].push(forklift_xy, wxy, audio_score, crane_active)

            # 오래 안 보이는 worker tracker 제거
            stale = [wid for wid, ts in last_seen.items()
                     if now - ts > EVICTION_SEC]
            for wid in stale:
                trackers.pop(wid, None)
                kinematics.pop(wid, None)
                last_seen.pop(wid, None)

            # ── 워커별 risk 예측 ──
            risks_per_worker: dict[str, np.ndarray] = {}
            for wid, tr in trackers.items():
                if tr.ready():
                    risks_per_worker[wid] = tr.predict()  # (1, 2)

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

                pred = FusionPrediction.from_model_output(risk)
                if not pred.has_alert(args.threshold):
                    continue

                direction = resolve_direction(
                    pred, args.threshold, kinematics[wid],
                    forklift_xy, dz_xy_for_alert,
                    forklift_speed=forklift_speed,
                    dz_force=dz_force,
                )
                if direction is None:
                    # 정지 forklift 거나 정면 충돌 등 — 알림 안 울림.
                    continue

                maybe_publish(
                    pred, args.threshold, direction, frame=snapshot_frame,
                )

            # 콘솔 로그 (200ms마다)
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
                    print(f"  - {wid} {wstr}  "
                          f"forklift_risk={risk[0,0]:.3f}  "
                          f"dropzone_risk={risk[0,1]:.3f}")
                    pred = FusionPrediction.from_model_output(risk)
                    for p in pred.triggered(args.threshold):
                        print(f"      🚨 {wid} ALERT → {p.threat_type.value} "
                              f"prob={p.prob:.3f} ({p.level.value})")

            # ── 시각화 (HEADLESS 면 통째로 skip) ──
            if not HEADLESS:
                bev_dz_xy = tuple(current_dz_center.tolist()) if has_live_dz else None
                bev_dz_r = current_dz_radius if has_live_dz else None
                bev_headings = {wid: k.heading for wid, k in kinematics.items()}
                bev = render_bev(
                    workers_xy, forklift_xy, audio_score, risks_per_worker,
                    threshold=args.threshold,
                    dropzone_xy=bev_dz_xy, dropzone_radius=bev_dz_r,
                    worker_headings=bev_headings,
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

    finally:
        audio_thread.stop()
        cam1.stop()
        if cam2:
            cam2.stop()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("\n[loop] 종료")


if __name__ == "__main__":
    main()
