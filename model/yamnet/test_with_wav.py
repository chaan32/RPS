"""WAV 파일로 모델 테스트 (학습과 동일한 전처리 적용).

peak normalize + frame 분할 + max pooling 으로 추론하여
학습/실시간/테스트의 결과가 일관되도록 한다.

사용법:
  python model/yamnet/test_with_wav.py <wav파일_또는_폴더>
"""

import os
import sys
import json
import glob

import numpy as np
import librosa
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CENTROID_PATH = os.path.join(MODEL_DIR, "anomaly_centroid.npy")
CONFIG_PATH = os.path.join(MODEL_DIR, "anomaly_config.json")

if len(sys.argv) < 2:
    print("사용법: python model/yamnet/test_with_wav.py <wav파일_또는_폴더>")
    sys.exit(1)

target = sys.argv[1]
if os.path.isdir(target):
    wav_files = sorted(glob.glob(os.path.join(target, "*.wav")))
elif os.path.isfile(target):
    wav_files = [target]
else:
    print(f"경로를 찾을 수 없습니다: {target}")
    sys.exit(1)

if not wav_files:
    print("wav 파일이 없습니다.")
    sys.exit(1)

centroid = np.load(CENTROID_PATH)
with open(CONFIG_PATH) as f:
    config = json.load(f)

THRESHOLD = config["threshold"]
FRAME_SEC = config.get("frame_sec", 0.96)
HOP_SEC = config.get("hop_sec", 0.48)
MIN_FRAME_RMS = config.get("min_frame_rms", 0.01)
SR = 16000

print("YAMNet 로드 중...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print(f"\n{'='*75}")
print(f"  임계값: {THRESHOLD:.4f}")
print(f"  전처리: peak normalize + frame 분할 + max pooling")
print(f"  파일: {len(wav_files)}개")
print(f"{'='*75}\n")


def peak_normalize(wav, target=0.95):
    peak = np.max(np.abs(wav))
    return wav if peak < 1e-6 else wav * (target / peak)


def split_to_frames(wav):
    frame_len = int(FRAME_SEC * SR)
    hop = int(HOP_SEC * SR)
    if len(wav) < frame_len:
        wav = np.pad(wav, (0, frame_len - len(wav)))
    frames = []
    for start in range(0, len(wav) - frame_len + 1, hop):
        chunk = wav[start:start + frame_len]
        if np.sqrt(np.mean(chunk**2)) >= MIN_FRAME_RMS:
            frames.append(chunk)
    if not frames:
        frames.append(wav[:frame_len])
    return frames


results = []
for path in wav_files:
    wav, _ = librosa.load(path, sr=SR, mono=True)
    wav = peak_normalize(wav.astype(np.float32))
    frames = split_to_frames(wav)

    sims = []
    for f in frames:
        _, embeddings, _ = yamnet_model(f.astype(np.float32))
        emb = embeddings.numpy()[0]
        s = float(cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0][0])
        sims.append(s)

    max_sim = max(sims) if sims else 0.0
    is_anomaly = max_sim >= THRESHOLD
    results.append((os.path.basename(path), max_sim, is_anomaly, len(frames)))

    bar_len = 30
    filled = int(max(0, max_sim) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    mark = "★ ANOMALY" if is_anomaly else "  normal "
    print(f"  [{bar}] max {max_sim:.3f}  frames={len(frames):>2}  {mark}  {os.path.basename(path)}")

detected = sum(1 for _, _, hit, _ in results if hit)
sims_all = [s for _, s, _, _ in results]
print(f"\n{'='*75}")
print(f"  요약: {detected}/{len(results)} 이상 감지 ({detected/len(results)*100:.1f}%)")
print(f"  max 유사도 — 평균 {np.mean(sims_all):.4f} | 최소 {np.min(sims_all):.4f} | 최대 {np.max(sims_all):.4f}")
print(f"{'='*75}")
