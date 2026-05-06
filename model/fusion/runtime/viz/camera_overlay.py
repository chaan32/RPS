"""카메라 프레임 위 detection + risk 오버레이.

cv2.imshow("cam1 + risk") 처럼 디버그 창에 띄우는 용도.
"""

from __future__ import annotations

import cv2

from .korean_text import put_korean, _get_korean_font


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
