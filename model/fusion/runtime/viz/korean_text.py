"""한글 텍스트 렌더링 (cv2.putText 는 한글 미지원).

PIL 로 한글 폰트 렌더 후 OpenCV BGR 로 다시 변환.
"""

from __future__ import annotations

import cv2
import numpy as np


_KOREAN_FONT_CACHE: dict = {}


def _get_korean_font(size: int):
    """Windows / macOS 에 흔한 한글 폰트 중 하나 로드 (캐시)."""
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
