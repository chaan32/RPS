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
DZ_SMOOTHING_FRAMES = 5                # 최근 N 프레임 median filter (시간 평활화)
MAX_WORKERS = 3                        # 동시 추적 가능한 작업자 수

# ── 작업자 운동학 (heading 추정 + 위협 상대 방향) ────
import math
from collections import deque as _deque


class WorkerKinematics:
    """worker별 위치 이력으로 heading(facing 방향) 추정.

    - 최근 N 프레임 위치 평균 속도 → heading 갱신 (속도 < 0.05m/s면 직전 값 유지)
    - default heading = π/2 (북쪽, +Y) — 검출 직후 첫 frame 처리용
    - collision_direction(threat_xy)로 worker body frame 기준 4방향(front/left/right/rear)
      반환
    """
    HISTORY_LEN = 5
    MIN_SPEED = 0.05    # m/frame 미만이면 정지 — heading 갱신 X

    def __init__(self):
        self.history = _deque(maxlen=self.HISTORY_LEN)
        self.heading = math.pi / 2     # 기본값: +Y (북쪽)
        self.last_xy = None

    def update(self, xy: tuple):
        self.history.append(xy)
        self.last_xy = xy
        if len(self.history) >= 3:
            pts = list(self.history)
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            mean_step = math.hypot(dx, dy) / max(1, len(pts) - 1)
            if mean_step > self.MIN_SPEED:
                self.heading = math.atan2(dy, dx)

    def collision_direction(self, threat_xy) -> str:
        """worker body frame 기준 위협 방향. 'front'|'left'|'right'|'rear'."""
        if self.last_xy is None:
            return "front"
        dx = threat_xy[0] - self.last_xy[0]
        dy = threat_xy[1] - self.last_xy[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return "front"
        threat_angle = math.atan2(dy, dx)
        rel = threat_angle - self.heading
        # [-π, π]로 정규화
        while rel > math.pi:
            rel -= 2 * math.pi
        while rel < -math.pi:
            rel += 2 * math.pi
        deg = math.degrees(rel)
        # body frame: 정면=0°, 좌측=+90°, 우측=-90°, 후방=±180°
        if -45 < deg <= 45:
            return "front"
        if 45 < deg <= 135:
            return "left"
        if -135 <= deg <= -45:
            return "right"
        return "rear"


def pick_positions(d1: list[dict], d2: list[dict]) -> tuple:
    """cam1 + cam2 detection list → (workers_xy, forklift_xy, dropzone_xy).

    Returns:
      workers_xy : dict {worker_id_str: (x, y)}  — ArUco 식별된 작업자만
      forklift_xy: tuple or None
      dropzone_xy: tuple or None  (box_1/box_2 = 인양물 평균 좌표)

    - worker  : ArUco worker_id 단위로 카메라 간 평균 (식별 못 된 worker는 제외)
    - forklift: 모든 카메라 평균
    - dropzone: box_1, box_2 (인양물) 평균. 없으면 None (이전 값 유지).
    """
    workers_by_id: dict[str, list] = {}
    forklifts, boxes = [], []
    for d in d1 + d2:
        t = d["type"]
        if t == "worker":
            wid = d.get("worker_id")
            if wid is None:
                # ArUco 식별 안 된 작업자는 무시 (W01/W02/W03 단위로 다루기 위해)
                continue
            workers_by_id.setdefault(wid, []).append(
                (d["world"]["x"], d["world"]["y"])
            )
        elif t == "forklift":
            forklifts.append((d["world"]["x"], d["world"]["y"]))
        elif t in BOX_CLASS_NAMES:
            boxes.append((d["world"]["x"], d["world"]["y"]))

    # worker_id별 cam1+cam2 평균
    workers_xy: dict[str, tuple[float, float]] = {}
    for wid, pts in workers_by_id.items():
        workers_xy[wid] = (
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
        dropzone_xy = (
            float(np.mean([p[0] for p in boxes])),
            float(np.mean([p[1] for p in boxes])),
        )
    return workers_xy, forklift_xy, dropzone_xy


# ── 한글 텍스트 렌더링 (cv2.putText는 한글 미지원) ───
_KOREAN_FONT_CACHE = {}

def _get_korean_font(size: int):
    """Windows에 흔한 한글 폰트 중 하나 로드 (캐시)."""
    if size in _KOREAN_FONT_CACHE:
        return _KOREAN_FONT_CACHE[size]
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
        "C:/Windows/Fonts/malgunbd.ttf",    # 맑은 고딕 Bold
        "C:/Windows/Fonts/gulim.ttc",       # 굴림
        "C:/Windows/Fonts/NanumGothic.ttf", # 나눔고딕
    ]
    font = None
    for fp in candidates:
        try:
            font = ImageFont.truetype(fp, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _KOREAN_FONT_CACHE[size] = font
    return font


def put_korean(img, text, position, font_size=20, color_bgr=(255, 255, 255)):
    """OpenCV BGR 이미지 위에 한글 텍스트 렌더링.

    position: (x, y) — 텍스트 좌상단 기준
    color_bgr: BGR 순서
    Returns: 새 BGR 이미지 (원본 미변경)
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # PIL 없으면 fallback (한글은 깨지지만 죽지는 않게)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 30.0, color_bgr, 2)
        return img
    font = _get_korean_font(font_size)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    pil_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text(position, text, font=font, fill=pil_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
    frame, detections, risks_per_worker, threshold=0.8,
):
    """카메라 프레임에 detection + risk 오버레이 (멀티 워커).

    Args:
      detections: world_pipeline 출력. worker는 d["worker_id"]로 식별됨
      risks_per_worker: dict {wid: (1, 2) ndarray} — wid별 risk_matrix
      threshold: 알림 발송 임계값
    """
    out = frame.copy()
    H, W = out.shape[:2]

    # detection 분류
    workers, threats = [], []
    for d in detections:
        if d["type"] == "worker":
            workers.append(d)
        elif d["type"] in ("forklift", "box_1", "box_2"):
            threats.append(d)

    # ── 위협 연결선 (worker별) ──
    for w in workers:
        wid = w.get("worker_id")
        if wid is None or wid not in risks_per_worker:
            continue
        risk = risks_per_worker[wid]
        f_risk = float(risk[0, 0])
        d_risk = float(risk[0, 1])
        wx1, wy1, wx2, wy2 = [int(v) for v in w["bbox_px"]]
        wcx, wcy = (wx1 + wx2) // 2, (wy1 + wy2) // 2
        for t in threats:
            tx1, ty1, tx2, ty2 = [int(v) for v in t["bbox_px"]]
            tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
            risk_for_line = f_risk if t["type"] == "forklift" else d_risk
            if risk_for_line >= 0.4:
                lc = _risk_color(risk_for_line, threshold)
                th = 3 if risk_for_line >= threshold else 2
                cv2.line(out, (wcx, wcy), (tcx, tcy), lc, th,
                         lineType=cv2.LINE_AA)
                mid = ((wcx + tcx) // 2, (wcy + tcy) // 2)
                txt = f"{wid}:{risk_for_line:.2f}"
                cv2.putText(out, txt, (mid[0] + 5, mid[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(out, txt, (mid[0] + 5, mid[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2)

    # ── Detection bbox + 라벨 ──
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_px"]]
        if d["type"] == "worker":
            wid = d.get("worker_id")
            if wid and wid in risks_per_worker:
                max_r = float(risks_per_worker[wid].max())
                c = _risk_color(max_r, threshold)
            else:
                c = _TYPE_COLORS["worker"]
            thickness = 3
            wx, wy = d["world"]["x"], d["world"]["y"]
            tag = wid if wid else "worker?"
            label = f'{tag} ({wx:.2f}, {wy:.2f})m'
        else:
            c = _TYPE_COLORS.get(d["type"], (200, 200, 200))
            thickness = 2
            wx, wy = d["world"]["x"], d["world"]["y"]
            label = f'{d["type"]} ({wx:.2f}, {wy:.2f})m'
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    # ── 우측 상단 Risk 패널 (per-worker rows) ──
    panel_w = 320
    panel_h = 60 + 36 * max(1, len(risks_per_worker))
    panel_h = min(panel_h, H - 20)
    px = W - panel_w - 10
    py = 10
    overlay = out.copy()
    cv2.rectangle(overlay, (px, py), (W - 10, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    cv2.rectangle(out, (px, py), (W - 10, py + panel_h), (255, 255, 255), 2)
    cv2.putText(out, "FUSION RISK (per worker)", (px + 10, py + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if risks_per_worker:
        row_y = py + 55
        for wid in sorted(risks_per_worker.keys()):
            risk = risks_per_worker[wid]
            f_r = float(risk[0, 0])
            d_r = float(risk[0, 1])
            f_c = _risk_color(f_r, threshold)
            d_c = _risk_color(d_r, threshold)
            cv2.putText(out,
                        f"{wid}: F={f_r:.2f}  DZ={d_r:.2f}",
                        (px + 10, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        f_c if f_r >= d_r else d_c, 2)
            row_y += 32
    else:
        cv2.putText(out, "(buffering...)", (px + 10, py + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ── 상단 ALERT 배너 ──
    if risks_per_worker:
        any_fork = any(float(r[0, 0]) >= threshold for r in risks_per_worker.values())
        any_dz = any(float(r[0, 1]) >= threshold for r in risks_per_worker.values())
        if any_fork or any_dz:
            if any_fork and any_dz:
                msg = "!! 지게차 충돌 + 인양물 위험 !!"
            elif any_fork:
                msg = "!! 지게차 충돌 위험 !!"
            else:
                msg = "!! 인양물 진입 위험 !!"
            cv2.rectangle(out, (0, 0), (W, 60), (0, 0, 255), -1)
            font = _get_korean_font(34)
            try:
                if hasattr(font, "getbbox"):
                    bbox = font.getbbox(msg)
                    text_w = bbox[2] - bbox[0]
                else:
                    text_w = font.getsize(msg)[0]
            except Exception:
                text_w = len(msg) * 22
            tx = max(20, (W - text_w) // 2)
            out = put_korean(out, msg, (tx, 12), 34, (255, 255, 255))

    return out


# ── BEV 시각화 ────────────────────────────────────────
def render_bev(
    workers_xy, forklift_xy, audio_score, risks_per_worker,
    threshold=DEFAULT_THRESHOLD,
    dropzone_xy=None, dropzone_radius=None,
    worker_headings=None,                      # dict {wid: heading_radians}
    view_bounds=((-4.0, 2.0), (-1.5, 4.5)),     # ArUco 사각형(2x3m)을 중앙에 + 주변 여백
    aruco_bounds=((-2.0, 0.0), (0.0, 3.0)),     # 실제 ArUco 사각형 (workspace)
    scale_px=120,
):
    """확장 BEV 평면도 + 멀티 워커 risk 게이지.

    Args:
      workers_xy: dict {wid: (x, y)}
      risks_per_worker: dict {wid: (1, 2) ndarray}
    """
    (x_min, x_max), (y_min, y_max) = view_bounds
    (ax_min, ax_max), (ay_min, ay_max) = aruco_bounds
    W = int((x_max - x_min) * scale_px) + 200    # +200: 우측 패널 공간
    H = int((y_max - y_min) * scale_px) + 100
    img = np.full((H, W, 3), 245, dtype=np.uint8)

    def w2px(wx, wy):
        px = int(50 + (wx - x_min) * scale_px)
        py = int(H - 50 - (wy - y_min) * scale_px)
        return px, py

    # 외곽 view 영역 박스 (전체 BEV 경계)
    p_view1 = w2px(x_min, y_min)
    p_view2 = w2px(x_max, y_max)
    cv2.rectangle(img, p_view1, p_view2, (200, 200, 200), 1)

    # 격자 (1m 간격) — 좌표 감 잡기
    for gx in range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1):
        p1 = w2px(gx, y_min)
        p2 = w2px(gx, y_max)
        cv2.line(img, p1, p2, (225, 225, 225), 1)
    for gy in range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1):
        p1 = w2px(x_min, gy)
        p2 = w2px(x_max, gy)
        cv2.line(img, p1, p2, (225, 225, 225), 1)

    # 원점 표시
    if x_min <= 0 <= x_max and y_min <= 0 <= y_max:
        ox, oy = w2px(0, 0)
        cv2.line(img, (ox - 10, oy), (ox + 10, oy), (180, 180, 180), 1)
        cv2.line(img, (ox, oy - 10), (ox, oy + 10), (180, 180, 180), 1)

    # ── ArUco 작업공간 (강조: 옅은 채움 + 굵은 테두리) ──
    p_a1 = w2px(ax_min, ay_min)
    p_a2 = w2px(ax_max, ay_max)
    overlay = img.copy()
    cv2.rectangle(overlay, p_a1, p_a2, (220, 235, 250), -1)   # 연한 파랑 채움
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, p_a1, p_a2, (80, 120, 180), 2)          # 진한 파랑 테두리
    # 라벨
    label_pt = (p_a1[0] + 5, p_a1[1] + 18)
    cv2.putText(img, "ArUco workspace (2m x 3m)", label_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 120, 180), 1)

    # ArUco 마커 4점 (실제 좌표 위치)
    for (mx, my, name) in [(-2, 0, "27"), (-2, 3, "22"), (0, 0, "38"), (0, 3, "24")]:
        px, py = w2px(mx, my)
        cv2.circle(img, (px, py), 8, (40, 80, 160), -1)
        cv2.circle(img, (px, py), 8, (255, 255, 255), 1)
        cv2.putText(img, name, (px - 28, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 80, 160), 1)

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

    # ── Workers (worker_id별 색상은 risk에 따라) ──
    if workers_xy:
        for wid in sorted(workers_xy.keys()):
            xy = workers_xy[wid]
            wpx = w2px(*xy)
            risk = risks_per_worker.get(wid)
            if risk is not None:
                max_r = float(risk.max())
                color = _risk_color(max_r, threshold)
            else:
                color = (0, 200, 0)
            cv2.circle(img, wpx, 13, color, -1)
            cv2.circle(img, wpx, 13, (255, 255, 255), 2)
            cv2.putText(img, wid, (wpx[0] - 22, wpx[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, wid, (wpx[0] - 22, wpx[1] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # heading 화살표 (작업자가 바라보는 방향)
            if worker_headings is not None and wid in worker_headings:
                hd = worker_headings[wid]
                # world heading vector → BEV pixel offset
                # world: x→오른쪽(+), y→위(+).  BEV에선 y축이 뒤집힘 (위쪽이 +y).
                arrow_len_px = 28
                end_px = (
                    int(wpx[0] + arrow_len_px * math.cos(hd)),
                    int(wpx[1] - arrow_len_px * math.sin(hd)),  # y 반전
                )
                cv2.arrowedLine(img, wpx, end_px, (0, 0, 0), 4,
                                tipLength=0.35, line_type=cv2.LINE_AA)
                cv2.arrowedLine(img, wpx, end_px, (255, 255, 255), 2,
                                tipLength=0.35, line_type=cv2.LINE_AA)

    # ── Forklift ──
    if forklift_xy is not None:
        fpx = w2px(*forklift_xy)
        cv2.rectangle(img, (fpx[0] - 15, fpx[1] - 10),
                      (fpx[0] + 15, fpx[1] + 10), (0, 0, 200), -1)
        cv2.putText(img, "F", (fpx[0] - 5, fpx[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Risk 패널 (오른쪽, per-worker rows) ──
    panel_x = W - 200
    cv2.rectangle(img, (panel_x, 20), (W - 20, H - 20), (255, 255, 255), -1)
    cv2.rectangle(img, (panel_x, 20), (W - 20, H - 20), (200, 200, 200), 2)
    cv2.putText(img, "RISK", (panel_x + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def gauge(label, value, y, color):
        cv2.putText(img, label, (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)
        cv2.rectangle(img, (panel_x + 10, y + 8),
                      (panel_x + 170, y + 22), (220, 220, 220), -1)
        v = float(np.clip(value, 0, 1))
        cv2.rectangle(img, (panel_x + 10, y + 8),
                      (panel_x + 10 + int(160 * v), y + 22), color, -1)
        # threshold 마커
        tx = panel_x + 10 + int(160 * threshold)
        cv2.line(img, (tx, y + 6), (tx, y + 24), (255, 255, 255), 1)
        cv2.putText(img, f"{v:.2f}", (panel_x + 130, y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    if risks_per_worker:
        row_y = 75
        for wid in sorted(risks_per_worker.keys()):
            risk = risks_per_worker[wid]
            f_r = float(risk[0, 0])
            d_r = float(risk[0, 1])
            cv2.putText(img, wid, (panel_x + 10, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            row_y += 22
            gauge("vs Forklift", f_r, row_y, _risk_color(f_r, threshold))
            row_y += 45
            gauge("vs DropZone", d_r, row_y, _risk_color(d_r, threshold))
            row_y += 55
    else:
        cv2.putText(img, "(buffering...)", (panel_x + 10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # 오디오 게이지 (항상 표시, 패널 하단)
    audio_y = H - 110
    gauge("Audio", audio_score, audio_y,
          (0, 0, 255) if audio_score >= 0.65 else
          (0, 200, 255) if audio_score >= 0.4 else (200, 200, 0))

    # ── 알림 배너 ──
    if risks_per_worker:
        any_alert = any(float(r.max()) >= threshold for r in risks_per_worker.values())
        if any_alert:
            cv2.rectangle(img, (panel_x + 5, H - 65),
                          (W - 25, H - 25), (0, 0, 255), -1)
            img = put_korean(img, "!! 위험 감지 !!",
                             (panel_x + 25, H - 60), 22, (255, 255, 255))

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
    print(f"[fusion] 모델 로드: {ckpt.name}")

    # worker별 독립 RealtimeInference 인스턴스 관리.
    # 같은 모델 객체를 공유하므로 메모리 부담 없음.
    trackers: dict[str, RealtimeInference] = {}
    last_seen: dict[str, float] = {}
    kinematics: dict[str, WorkerKinematics] = {}
    EVICTION_SEC = 2.0   # N초 이상 미감지 시 tracker 제거

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

    # 인양물 BEV 좌표 시간 평활화 버퍼 (최근 N 프레임 median)
    from collections import deque
    dz_history = deque(maxlen=DZ_SMOOTHING_FRAMES)

    # 현재 적용 중인 dropzone (live 모드 아니면 default)
    import numpy as _np_main
    current_dz_center = _np_main.array(DZ_CENTER, dtype=_np_main.float32)
    current_dz_radius = float(DZ_RADIUS)

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

            workers_xy, forklift_xy, dropzone_xy = pick_positions(d1, d2)

            with _audio_lock:
                audio_score = _audio_state["score"]

            # 인양물 검출되면 dz 좌표 갱신. 공중 객체는 homography 오차가 크므로
            # 최근 N 프레임 median 필터로 안정화.
            if dropzone_xy is not None:
                dz_history.append(dropzone_xy)
                xs = sorted(p[0] for p in dz_history)
                ys = sorted(p[1] for p in dz_history)
                mid = len(xs) // 2
                smoothed_dz = (xs[mid], ys[mid])
                current_dz_center = _np_main.array(smoothed_dz, dtype=_np_main.float32)
                has_live_dz = True
                # 모든 활성 tracker에 전파
                for tr in trackers.values():
                    tr.update_dropzone(center=smoothed_dz)

            crane_active = 0  # TODO: MQTT crane state

            # 검출된 worker마다 tracker + kinematics 생성/갱신
            for wid, xy in workers_xy.items():
                last_seen[wid] = now
                if wid not in trackers:
                    if len(trackers) >= MAX_WORKERS:
                        continue   # 최대 추적 수 초과
                    new_tr = RealtimeInference(model, device="cpu")
                    if has_live_dz:
                        new_tr.update_dropzone(
                            center=tuple(current_dz_center.tolist())
                        )
                    trackers[wid] = new_tr
                if wid not in kinematics:
                    kinematics[wid] = WorkerKinematics()
                kinematics[wid].update(xy)
                trackers[wid].push(forklift_xy, xy, audio_score, crane_active)

            # 미감지 worker tracker 제거 (kinematics도 같이)
            for wid in list(trackers.keys()):
                if now - last_seen.get(wid, 0) > EVICTION_SEC:
                    del trackers[wid]
                    last_seen.pop(wid, None)
                    kinematics.pop(wid, None)

            # 각 worker별 risk 추론
            risks_per_worker: dict[str, "_np_main.ndarray"] = {}
            for wid, tr in trackers.items():
                if tr.ready():
                    risks_per_worker[wid] = tr.predict()    # (1, 2)

            # ── 콘솔 로그 (200ms마다, 워커당 한 줄 형식) ──
            if now - last_print >= print_period:
                last_print = now
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                f_str = (f"({forklift_xy[0]:+.2f},{forklift_xy[1]:+.2f})"
                         if forklift_xy else "(none)")
                dz_tag = "live" if has_live_dz else "default"
                dz_str = (f"({float(current_dz_center[0]):+.2f},"
                          f"{float(current_dz_center[1]):+.2f}|{dz_tag})")

                if not workers_xy:
                    # 워커 미감지: risk는 0.000 으로 표기
                    print(
                        f"t={ts}  audio={audio_score:.2f}  "
                        f"W=(none)  F={f_str}  DZ={dz_str}  "
                        f"forklift_risk=0.000  dropzone_risk=0.000"
                    )
                else:
                    # 워커별 한 줄: t, audio, W{ID}, F, DZ, forklift_risk, dropzone_risk
                    for wid in sorted(workers_xy.keys()):
                        xy = workers_xy[wid]
                        w_str = f"({xy[0]:+.2f},{xy[1]:+.2f})"
                        risk = risks_per_worker.get(wid)
                        if risk is not None:
                            f_r = float(risk[0, 0])
                            d_r = float(risk[0, 1])
                            risk_str = (
                                f"forklift_risk={f_r:.3f}  "
                                f"dropzone_risk={d_r:.3f}"
                            )
                        else:
                            risk_str = "forklift_risk=---    dropzone_risk=---    (buffering)"
                        print(
                            f"t={ts}  audio={audio_score:.2f}  "
                            f"{wid}={w_str}  F={f_str}  DZ={dz_str}  {risk_str}"
                        )

                # 알림 라인 (per worker, per threat).
                # forklift는 worker body frame 기준 방향(left/right/rear/front) 계산,
                # dropzone은 항상 'all' (전방향 진동).
                for wid in sorted(risks_per_worker.keys()):
                    risk = risks_per_worker[wid]
                    f_r = float(risk[0, 0])
                    d_r = float(risk[0, 1])
                    if f_r >= args.threshold:
                        if forklift_xy is not None and wid in kinematics:
                            direction = kinematics[wid].collision_direction(forklift_xy)
                        else:
                            direction = "front"
                        print(
                            f"  🚨 ALERT → worker/{wid}/vibration  "
                            f"scenario=forklift  dir={direction}  prob={f_r:.3f}"
                        )
                    if d_r >= args.threshold:
                        print(
                            f"  🚨 ALERT → worker/{wid}/vibration  "
                            f"scenario=dropzone  dir=all  prob={d_r:.3f}"
                        )

            # ── 시각화 ──
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
