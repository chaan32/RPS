"""실시간 마이크 이상 소리 감지.

노트북 학습과 동일한 파이프라인을 적용:
  마이크 buffer → peak normalize → 0.96s frame 분할 → YAMNet 임베딩 → centroid 유사도 → max pooling

실행: python input/audio/realtime_detect.py
종료: Ctrl+C
"""

import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


# ── 경로 & 사용자 설정 ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "yamnet")
CENTROID_PATH = os.path.join(MODEL_DIR, "anomaly_centroid.npy")
CONFIG_PATH = os.path.join(MODEL_DIR, "anomaly_config.json")

THRESHOLD_OVERRIDE = None     # None이면 config의 threshold 사용, 값 지정 시 덮어쓰기
MIN_RMS = 0.003               # 원본 버퍼 RMS가 이보다 낮으면 분석 스킵
BUFFER_SEC = 1.92             # 1회 분석할 오디오 길이 (초)


# ── 모델/설정 로드 ────────────────────────────────────────────────────
def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


if not os.path.exists(CENTROID_PATH):
    fail(f"centroid 파일 없음: {CENTROID_PATH}\n  먼저 model/yamnet/yamnet_transfer_learning.ipynb 실행")

centroid = np.load(CENTROID_PATH)
with open(CONFIG_PATH) as f:
    config = json.load(f)

THRESHOLD = THRESHOLD_OVERRIDE if THRESHOLD_OVERRIDE is not None else config["threshold"]
FRAME_SEC = config.get("frame_sec", 0.96)
HOP_SEC = config.get("hop_sec", 0.48)
MIN_FRAME_RMS = config.get("min_frame_rms", 0.01)

SAMPLE_RATE = 16000
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SEC)
FRAME_LEN = int(FRAME_SEC * SAMPLE_RATE)
HOP_LEN = int(HOP_SEC * SAMPLE_RATE)

print(f"\n{'='*65}")
print(f"  실시간 이상 소리 감지")
print(f"{'='*65}")
print(f"  학습 데이터: {config.get('num_files','?')}파일 / {config.get('num_frames','?')}프레임 "
      f"({config.get('source','?')})")
print(f"  CV Test Recall: {config.get('cv_test_recall','?')}")
print(f"  임계값: {THRESHOLD:.3f}  (>={THRESHOLD:.3f} 면 이상 감지)")
print(f"  버퍼: {BUFFER_SEC}s / 분석 프레임: {FRAME_SEC}s")
print(f"  무음 스킵: RMS < {MIN_RMS}")
print(f"{'='*65}")

print("\nYAMNet 로드 중...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet 준비 완료!\n")


# ── 전처리 함수 (학습과 동일) ──────────────────────────────────────────
def peak_normalize(wav, target=0.95):
    peak = float(np.max(np.abs(wav)))
    return wav if peak < 1e-6 else wav * (target / peak)


def split_to_frames(wav):
    if len(wav) < FRAME_LEN:
        wav = np.pad(wav, (0, FRAME_LEN - len(wav)))
    frames = []
    for start in range(0, len(wav) - FRAME_LEN + 1, HOP_LEN):
        chunk = wav[start:start + FRAME_LEN]
        if np.sqrt(np.mean(chunk ** 2)) >= MIN_FRAME_RMS:
            frames.append(chunk)
    return frames if frames else [wav[:FRAME_LEN]]


def compute_max_similarity(wav):
    """peak normalize → frame 분할 → 각 frame 유사도 → max."""
    wav = peak_normalize(wav)
    frames = split_to_frames(wav)
    sims = []
    for frame in frames:
        _, embeddings, _ = yamnet_model(frame.astype(np.float32))
        emb = embeddings.numpy()[0]
        sims.append(float(cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0][0]))
    return max(sims), len(frames)


def render_bar(value, width=30):
    filled = int(max(0, min(1, value)) * width)
    return "█" * filled + "░" * (width - filled)


# ── 실시간 루프 ───────────────────────────────────────────────────────
print(f"마이크 감지 시작 (Ctrl+C로 종료)\n")

try:
    while True:
        audio = sd.rec(BUFFER_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        wav = audio.flatten().astype(np.float32)

        rms = float(np.sqrt(np.mean(wav ** 2)))
        if rms < MIN_RMS:
            print(f"  [조용함 {render_bar(0)}]                    RMS {rms:.4f}")
            continue

        max_sim, n_frames = compute_max_similarity(wav)
        bar = render_bar(max_sim)

        if max_sim >= THRESHOLD:
            print(f"  [{bar}] max {max_sim:.3f}  frames={n_frames}  RMS {rms:.4f}  *** 이상 감지! ***")
        else:
            print(f"  [{bar}] max {max_sim:.3f}  frames={n_frames}  RMS {rms:.4f}")

except KeyboardInterrupt:
    print("\n\n종료되었습니다.")
