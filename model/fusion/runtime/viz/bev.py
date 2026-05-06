"""BEV (Bird's-Eye View) 평면도 시각화.

월드 좌표계를 위에서 내려다본 평면도로 변환 + per-worker risk 게이지.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ...inference import DEFAULT_THRESHOLD
from ...data.scenario_generator import DZ_CENTER, DZ_RADIUS
from .korean_text import put_korean
from .camera_overlay import _risk_color


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
