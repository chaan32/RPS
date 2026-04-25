"""실시간 카메라 + YAMnet → Fusion 모델 통합 검증.

데이터 흐름:
  [Cam1 (RTSP)] ──► YOLO+ArUco+homography ──► world coords (worker, forklift)
  [Cam2 (RTSP)] ─┘                                   │
                                                      ▼
  [로컬 마이크 (background thread)] ──► YAMnet ──► audio_score
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
  python model/fusion/realtime_camera.py            # 라이브 카메라
  python model/fusion/realtime_camera.py --no-audio # YAMnet 끄고 카메라만

ESC: 종료
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import threading
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch

# RTSP TCP 강제 (cv2가 import되기 전에 설정)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|analyzeduration;0|probesize;32",
)

# 프로젝트 경로
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE.parent))                     # model/fusion
sys.path.insert(0, str(PROJECT_ROOT / "input" / "media")) # world_pipeline, camera

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Fusion 모듈
from inference import (
    load_model,
    RealtimeInference,
    risk_matrix_to_json,
    build_alerts,
    DEFAULT_THRESHOLD,
)
from scenario_generator import DZ_CENTER, DZ_RADIUS, RATE


# ── 공유 상태 (audio thread <-> main loop) ─────────────
_audio_state = {"score": 0.05, "ts": 0.0}
_audio_lock = threading.Lock()
_running = True


# ── YAMnet 백그라운드 오디오 스레드 ────────────────────
def audio_worker(verbose: bool = False) -> None:
    """로컬 마이크 → YAMnet → audio score 갱신 (1.92s buffer)."""
    global _running
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        import sounddevice as sd
        import tensorflow_hub as hub
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        print(f"[audio] 패키지 누락: {e}. --no-audio 옵션으로 끄세요.")
        return

    yamnet_dir = PROJECT_ROOT / "model" / "yamnet"
    centroid_path = yamnet_dir / "anomaly_centroid.npy"
    config_path = yamnet_dir / "anomaly_config.json"
    if not centroid_path.exists():
        print(f"[audio] centroid 없음: {centroid_path}")
        return

    centroid = np.load(centroid_path)
    with open(config_path) as f:
        cfg = json.load(f)

    SAMPLE_RATE = 16000
    BUFFER_SEC = 1.92
    FRAME_SEC = cfg.get("frame_sec", 0.96)
    HOP_SEC = cfg.get("hop_sec", 0.48)
    MIN_FRAME_RMS = cfg.get("min_frame_rms", 0.01)
    BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SEC)
    FRAME_LEN = int(FRAME_SEC * SAMPLE_RATE)
    HOP_LEN = int(HOP_SEC * SAMPLE_RATE)

    print("[audio] YAMnet 로드 중...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("[audio] YAMnet 준비 완료, 마이크 시작")

    while _running:
        try:
            audio = sd.rec(BUFFER_SIZE, samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32")
            sd.wait()
        except Exception as e:
            print(f"[audio] 마이크 오류: {e}")
            time.sleep(1.0)
            continue

        wav = audio.flatten().astype(np.float32)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        if rms < 0.003:
            score = 0.0
        else:
            # peak normalize + frame split + cosine similarity
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-6:
                wav = wav * (0.95 / peak)
            if len(wav) < FRAME_LEN:
                wav = np.pad(wav, (0, FRAME_LEN - len(wav)))
            sims = []
            for start in range(0, len(wav) - FRAME_LEN + 1, HOP_LEN):
                chunk = wav[start:start + FRAME_LEN]
                if np.sqrt(np.mean(chunk ** 2)) < MIN_FRAME_RMS:
                    continue
                _, embeddings, _ = yamnet_model(chunk.astype(np.float32))
                emb = embeddings.numpy()[0]
                sim = float(cosine_similarity(
                    emb.reshape(1, -1), centroid.reshape(1, -1)
                )[0][0])
                sims.append(sim)
            score = max(sims) if sims else 0.0

        with _audio_lock:
            _audio_state["score"] = float(np.clip(score, 0.0, 1.0))
            _audio_state["ts"] = time.time()

        if verbose:
            print(f"[audio] max_sim={score:.3f}  rms={rms:.4f}")


# ── 카메라 detection 병합 ──────────────────────────────
BOX_CLASS_NAMES = ("box_1", "box_2")   # 크레인 인양물 (= 동적 dropzone 위치)


def pick_positions(d1: list[dict], d2: list[dict]) -> tuple:
    """cam1 + cam2 detection list → (worker_xy, forklift_xy, dropzone_xy).

    - worker  : ArUco track_id 매칭으로 카메라 간 평균
    - forklift: 모든 카메라 평균
    - dropzone: box_1, box_2 (인양물) 평균. 없으면 None (이전 값 유지).
    """
    workers, forklifts, boxes = {}, [], []
    for d in d1 + d2:
        t = d["type"]
        if t == "worker":
            tid = d.get("track_id")
            if tid is None:
                tid = f"_anon_{len(workers)}"
            workers.setdefault(tid, []).append((d["world"]["x"], d["world"]["y"]))
        elif t == "forklift":
            forklifts.append((d["world"]["x"], d["world"]["y"]))
        elif t in BOX_CLASS_NAMES:
            boxes.append((d["world"]["x"], d["world"]["y"]))

    worker_xy = None
    if workers:
        first_tid = list(workers.keys())[0]
        pts = workers[first_tid]
        worker_xy = (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
        )

    forklift_xy = None
    if forklifts:
        forklift_xy = (
            float(np.mean([p[0] for p in forklifts])),
            float(np.mean([p[1] for p in forklifts])),
        )

    dropzone_xy = None
    if boxes:
        # 다중 인양물이면 평균. 단일 부하면 그대로.
        dropzone_xy = (
            float(np.mean([p[0] for p in boxes])),
            float(np.mean([p[1] for p in boxes])),
        )
    return worker_xy, forklift_xy, dropzone_xy


# ── 카메라 프레임 위 risk 오버레이 ───────────────────
_TYPE_COLORS = {
    "worker":   (0, 255, 255),  # 노랑 (default)
    "forklift": (0, 0, 255),    # 빨강
    "box_1":    (200, 0, 200),  # 자주 (인양물)
    "box_2":    (200, 0, 200),
}


def _risk_color(value: float, threshold: float):
    """risk 값에 따른 색상 (BGR)."""
    if value >= threshold:
        return (0, 0, 255)        # 빨강 (danger)
    if value >= 0.4:
        return (0, 165, 255)      # 주황 (warning)
    return (0, 200, 0)            # 초록 (safe)


def draw_camera_overlay(
    frame, detections, risk_matrix, threshold=0.8,
):
    """카메라 프레임에 detection + risk 오버레이.

    - bbox 색: worker는 risk에 따라, 다른 객체는 type 기반
    - 우측 상단 패널: vs_Forklift / vs_DropZone 게이지
    - 상단 ALERT 배너: max risk ≥ threshold 시
    - 작업자 ↔ 위협 연결선: 같은 카메라 프레임 내에 둘 다 보일 때
    """
    out = frame.copy()
    H, W = out.shape[:2]

    # worker bbox 색 결정
    if risk_matrix is not None:
        max_risk = float(risk_matrix.max())
        worker_color = _risk_color(max_risk, threshold)
    else:
        worker_color = _TYPE_COLORS["worker"]

    # detection 분류 (선 그릴 때 위치 참조)
    workers, threats = [], []
    for d in detections:
        if d["type"] == "worker":
            workers.append(d)
        elif d["type"] in ("forklift", "box_1", "box_2"):
            threats.append(d)

    # ── 위협 연결선 (worker가 있고 risk가 warn 이상일 때) ──
    if risk_matrix is not None and workers:
        f_risk = float(risk_matrix[0, 0])
        d_risk = float(risk_matrix[0, 1])
        for w in workers:
            wx1, wy1, wx2, wy2 = [int(v) for v in w["bbox_px"]]
            wcx, wcy = (wx1 + wx2) // 2, (wy1 + wy2) // 2
            for t in threats:
                tx1, ty1, tx2, ty2 = [int(v) for v in t["bbox_px"]]
                tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                if t["type"] == "forklift":
                    risk_for_line = f_risk
                else:
                    risk_for_line = d_risk
                if risk_for_line >= 0.4:
                    line_color = _risk_color(risk_for_line, threshold)
                    line_thickness = 3 if risk_for_line >= threshold else 2
                    cv2.line(out, (wcx, wcy), (tcx, tcy),
                             line_color, line_thickness, lineType=cv2.LINE_AA)
                    # 거리 표시
                    mid = ((wcx + tcx) // 2, (wcy + tcy) // 2)
                    cv2.putText(out, f"{risk_for_line:.2f}",
                                (mid[0] + 5, mid[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 0), 4)
                    cv2.putText(out, f"{risk_for_line:.2f}",
                                (mid[0] + 5, mid[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                line_color, 2)

    # ── Detection bbox + 라벨 ──
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_px"]]
        if d["type"] == "worker":
            c = worker_color
        else:
            c = _TYPE_COLORS.get(d["type"], (200, 200, 200))
        thickness = 3 if d["type"] == "worker" else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)

        wx, wy = d["world"]["x"], d["world"]["y"]
        label = f'{d["type"]} ({wx:.2f}, {wy:.2f})m'
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    # ── 우측 상단 Risk 패널 ──
    panel_w = 300
    panel_h = 180
    px = W - panel_w - 10
    py = 10
    overlay = out.copy()
    cv2.rectangle(overlay, (px, py), (W - 10, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    cv2.rectangle(out, (px, py), (W - 10, py + panel_h), (255, 255, 255), 2)
    cv2.putText(out, "FUSION RISK", (px + 10, py + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if risk_matrix is not None:
        f_risk = float(risk_matrix[0, 0])
        d_risk = float(risk_matrix[0, 1])

        def gauge(label, value, y):
            cv2.putText(out, f"{label}: {value:.2f}", (px + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _risk_color(value, threshold), 2)
            bar_x = px + 10
            bar_y = y + 8
            bar_w_max = panel_w - 30
            cv2.rectangle(out, (bar_x, bar_y),
                          (bar_x + bar_w_max, bar_y + 12),
                          (80, 80, 80), -1)
            v = max(0.0, min(1.0, value))
            cv2.rectangle(out, (bar_x, bar_y),
                          (bar_x + int(bar_w_max * v), bar_y + 12),
                          _risk_color(value, threshold), -1)
            # threshold 표시 마커
            tx = bar_x + int(bar_w_max * threshold)
            cv2.line(out, (tx, bar_y - 2), (tx, bar_y + 14), (255, 255, 255), 1)

        gauge("vs Forklift", f_risk, py + 65)
        gauge("vs DropZone", d_risk, py + 130)
    else:
        cv2.putText(out, "(buffering...)", (px + 10, py + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── 상단 ALERT 배너 ──
    if risk_matrix is not None:
        f_risk = float(risk_matrix[0, 0])
        d_risk = float(risk_matrix[0, 1])
        max_risk = max(f_risk, d_risk)
        if max_risk >= threshold:
            if f_risk >= threshold and d_risk >= threshold:
                msg = "!! COLLISION + DROPZONE !!"
            elif f_risk >= threshold:
                msg = "!! FORKLIFT COLLISION !!"
            else:
                msg = "!! DROPZONE INTRUSION !!"
            cv2.rectangle(out, (0, 0), (W, 60), (0, 0, 255), -1)
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            tx = max(20, (W - text_size[0]) // 2)
            cv2.putText(out, msg, (tx, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 3)

    return out


# ── BEV 시각화 ────────────────────────────────────────
def render_bev(
    worker_xy, forklift_xy, audio_score, risk_matrix,
    dropzone_xy=None, dropzone_radius=None,
    workspace=((-2.0, 0.0), (0.0, 3.0)),
    scale_px=180,
):
    """작업공간 평면도 + 실시간 risk 게이지."""
    (x_min, x_max), (y_min, y_max) = workspace
    W = int((x_max - x_min) * scale_px) + 200
    H = int((y_max - y_min) * scale_px) + 100
    img = np.full((H, W, 3), 240, dtype=np.uint8)

    def w2px(wx, wy):
        # wx ∈ [x_min, x_max] → px ∈ [50, 50 + (x_max-x_min)*scale_px]
        # wy ∈ [y_min, y_max] → py ∈ [H-50 - (y_max-y_min)*scale_px, H-50]  (Y 뒤집힘)
        px = int(50 + (wx - x_min) * scale_px)
        py = int(H - 50 - (wy - y_min) * scale_px)
        return px, py

    # 작업공간 박스
    p1 = w2px(x_min, y_min)
    p2 = w2px(x_max, y_max)
    cv2.rectangle(img, p1, p2, (180, 180, 180), 2)

    # ArUco 마커 4점
    for (mx, my, name) in [(-2, 0, "27"), (-2, 3, "22"), (0, 0, "38"), (0, 3, "24")]:
        px, py = w2px(mx, my)
        cv2.circle(img, (px, py), 7, (60, 60, 60), -1)
        cv2.putText(img, name, (px - 25, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    # Dropzone (동적 위치 우선, 없으면 학습 default)
    dz_cx, dz_cy = (dropzone_xy if dropzone_xy is not None
                     else (DZ_CENTER[0], DZ_CENTER[1]))
    dz_r = dropzone_radius if dropzone_radius is not None else DZ_RADIUS
    dz_px = w2px(dz_cx, dz_cy)
    dz_color = (0, 100, 200) if dropzone_xy is None else (0, 50, 255)  # live는 진한 빨강
    cv2.circle(img, dz_px, int(dz_r * scale_px), dz_color, 2)
    cv2.putText(img, "DZ" if dropzone_xy is None else "DZ(live)",
                (dz_px[0] - 25, dz_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, dz_color, 2)

    # Worker
    if worker_xy is not None:
        wpx = w2px(*worker_xy)
        cv2.circle(img, wpx, 12, (0, 200, 0), -1)
        cv2.putText(img, "W", (wpx[0] - 6, wpx[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Forklift
    if forklift_xy is not None:
        fpx = w2px(*forklift_xy)
        cv2.rectangle(img, (fpx[0] - 15, fpx[1] - 10),
                      (fpx[0] + 15, fpx[1] + 10), (0, 0, 200), -1)
        cv2.putText(img, "F", (fpx[0] - 5, fpx[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Risk 게이지 (오른쪽)
    panel_x = W - 180
    cv2.rectangle(img, (panel_x, 20), (W - 20, H - 20), (255, 255, 255), -1)
    cv2.rectangle(img, (panel_x, 20), (W - 20, H - 20), (200, 200, 200), 2)
    cv2.putText(img, "RISK", (panel_x + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def gauge(label, value, y, color_lo, color_hi):
        cv2.putText(img, label, (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.rectangle(img, (panel_x + 10, y + 10),
                      (panel_x + 150, y + 30), (220, 220, 220), -1)
        v = float(np.clip(value, 0, 1))
        bar_w = int(140 * v)
        # 색: 파랑 → 빨강
        if v > 0.8:
            c = (0, 0, 255)
        elif v > 0.4:
            c = (0, 200, 255)
        else:
            c = (200, 200, 0)
        cv2.rectangle(img, (panel_x + 10, y + 10),
                      (panel_x + 10 + bar_w, y + 30), c, -1)
        cv2.putText(img, f"{v:.2f}", (panel_x + 100, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if risk_matrix is not None:
        gauge("vs Forklift", risk_matrix[0, 0], 80, "lo", "hi")
        gauge("vs DropZone", risk_matrix[0, 1], 160, "lo", "hi")
    else:
        cv2.putText(img, "(buffering...)", (panel_x + 10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    gauge("Audio", audio_score, 240, "lo", "hi")

    # 알림 배너
    if risk_matrix is not None:
        max_risk = float(risk_matrix.max())
        if max_risk >= DEFAULT_THRESHOLD:
            cv2.rectangle(img, (panel_x + 5, H - 70),
                          (W - 25, H - 30), (0, 0, 255), -1)
            cv2.putText(img, "!! ALERT !!", (panel_x + 20, H - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


# ── 메인 ─────────────────────────────────────────────
def main():
    global _running

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

    # ── 모델 로드 ──
    ckpt = _HERE.parent / "checkpoints" / "best.pt"
    if not ckpt.exists():
        print(f"체크포인트 없음: {ckpt}")
        sys.exit(1)
    model = load_model(str(ckpt), device="cpu")
    rt = RealtimeInference(model, device="cpu")
    print(f"[fusion] 모델 로드: {ckpt.name}")

    # ── 카메라 ──
    from camera import VideoStream
    from world_pipeline import extract_detections_with_world, ensure_calibration

    rtsp1 = os.getenv("CAMERA_RTSP_URL_1")
    rtsp2 = os.getenv("CAMERA_RTSP_URL_2")
    if not rtsp1:
        print(".env CAMERA_RTSP_URL_1 필요")
        sys.exit(1)

    print("[cam] 캘리브레이션 확인 중...")
    ensure_calibration("cam1")
    if not args.no_cam2:
        ensure_calibration("cam2")

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
    audio_thread = None
    if not args.no_audio:
        audio_thread = threading.Thread(target=audio_worker, daemon=True)
        audio_thread.start()

    # ── 메인 루프 (5 FPS) ──
    print("\n[loop] 시작 (ESC 종료)\n")
    last_iter = 0.0
    iter_period = 1.0 / RATE   # 0.2s
    print_period = 0.2
    last_print = 0.0
    has_live_dz = False           # 인양물 검출되어 dropzone 갱신된 적 있는지

    try:
        while _running:
            now = time.time()
            if now - last_iter < iter_period:
                time.sleep(0.005)
                continue
            last_iter = now

            ret1, f1 = cam1.read()
            ret2, f2 = (cam2.read() if cam2 else (False, None))
            d1 = extract_detections_with_world(f1, "cam1") if ret1 and f1 is not None else []
            d2 = extract_detections_with_world(f2, "cam2") if ret2 and f2 is not None else []

            worker_xy, forklift_xy, dropzone_xy = pick_positions(d1, d2)

            with _audio_lock:
                audio_score = _audio_state["score"]

            # 인양물 검출되면 dropzone 좌표 갱신 (없으면 직전 값 유지)
            if dropzone_xy is not None:
                rt.update_dropzone(center=dropzone_xy)
                has_live_dz = True

            # buffer push (worker가 없으면 기본값으로 멀리 둠 → safe)
            push_worker = worker_xy if worker_xy else (-0.3, 5.0)
            crane_active = 0  # TODO: MQTT crane state
            rt.push(forklift_xy, push_worker, audio_score, crane_active)

            risk = None
            if rt.ready():
                risk = rt.predict()  # (1, 2)

                # 콘솔 로그 (200ms마다)
                if now - last_print >= print_period:
                    last_print = now
                    line = (f"t={datetime.now().strftime('%H:%M:%S.%f')[:-3]}  "
                            f"audio={audio_score:.2f}  ")
                    if worker_xy:
                        line += f"W=({worker_xy[0]:+.2f},{worker_xy[1]:+.2f})  "
                    else:
                        line += f"W=(none)              "
                    if forklift_xy:
                        line += f"F=({forklift_xy[0]:+.2f},{forklift_xy[1]:+.2f})  "
                    else:
                        line += f"F=(none)              "
                    line += (f"forklift_risk={risk[0,0]:.3f}  "
                            f"dropzone_risk={risk[0,1]:.3f}")
                    print(line)

                    # 알림
                    if (risk[0, 0] >= args.threshold or risk[0, 1] >= args.threshold):
                        json_pred = risk_matrix_to_json(risk)
                        for a in build_alerts(json_pred, threshold=args.threshold):
                            print(f"  🚨 ALERT → {a['topic']}  "
                                  f"scenario={a['scenario']}  "
                                  f"dir={a['direction']}  "
                                  f"prob={a['prob']:.3f}")

            # ── 시각화 ──
            # live dropzone이 들어왔으면 해당 좌표를 BEV에 빨간색으로 표시,
            # 아니면 학습 default(-1.5, 2.0)를 회색으로
            current_dz = tuple(rt._dropzone_center.tolist()) if has_live_dz else None
            current_dz_r = float(rt._dropzone_radius) if has_live_dz else None
            bev = render_bev(
                worker_xy, forklift_xy, audio_score, risk,
                dropzone_xy=current_dz, dropzone_radius=current_dz_r,
            )
            cv2.imshow("Fusion BEV", bev)

            if not args.no_frames:
                if ret1 and f1 is not None:
                    overlay1 = draw_camera_overlay(f1, d1, risk, args.threshold)
                    cv2.imshow("cam1 + risk", cv2.resize(overlay1, (900, 600)))
                if ret2 and f2 is not None:
                    overlay2 = draw_camera_overlay(f2, d2, risk, args.threshold)
                    cv2.imshow("cam2 + risk", cv2.resize(overlay2, (900, 600)))

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        _running = False
        cam1.stop()
        if cam2:
            cam2.stop()
        cv2.destroyAllWindows()
        print("\n[loop] 종료")


if __name__ == "__main__":
    main()
