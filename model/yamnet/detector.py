"""YAMNet + centroid 기반 이상 소리 탐지기.

realtime_detect.py / test_with_wav.py / WebSocket 서버가 공통으로 쓸 수 있도록
학습 파이프라인(peak normalize → frame 분할 → YAMNet 임베딩 → centroid 코사인 유사도 → max)을
클래스로 묶어 둔 모듈.

모델 로드는 YamnetDetector() 한 번에 끝나며 이후 predict(wav) 호출은 같은 인스턴스를 재사용한다.
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # ERROR만 (cpu_feature_guard 등 억제)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CENTROID_PATH = os.path.join(MODEL_DIR, "anomaly_centroid.npy")
DEFAULT_CONFIG_PATH = os.path.join(MODEL_DIR, "anomaly_config.json")
YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"


class YamnetDetector:
    """YAMNet 임베딩과 centroid 유사도로 이상 소리를 판정한다.

    Parameters
    ----------
    centroid_path, config_path:
        학습 노트북(yamnet_transfer_learning.ipynb)에서 만든 파일 경로.
        None이면 model/yamnet/ 기본 경로를 사용.
    threshold_override:
        config의 threshold를 덮어쓰고 싶을 때 지정.
    """

    def __init__(self, centroid_path=None, config_path=None, threshold_override=None):
        self.centroid = np.load(centroid_path or DEFAULT_CENTROID_PATH)
        with open(config_path or DEFAULT_CONFIG_PATH) as f:
            self.config = json.load(f)

        self.threshold = (
            threshold_override if threshold_override is not None else self.config["threshold"]
        )
        self.frame_sec = self.config.get("frame_sec", 0.96)
        self.hop_sec = self.config.get("hop_sec", 0.48)
        self.min_frame_rms = self.config.get("min_frame_rms", 0.01)

        self.sample_rate = 16000
        self.frame_len = int(self.frame_sec * self.sample_rate)
        self.hop_len = int(self.hop_sec * self.sample_rate)

        self.yamnet = hub.load(YAMNET_HUB_URL)

    @staticmethod
    def peak_normalize(wav, target=0.95):
        peak = float(np.max(np.abs(wav)))
        return wav if peak < 1e-6 else wav * (target / peak)

    def _split_frames(self, wav):
        if len(wav) < self.frame_len:
            wav = np.pad(wav, (0, self.frame_len - len(wav)))
        frames = []
        for start in range(0, len(wav) - self.frame_len + 1, self.hop_len):
            chunk = wav[start:start + self.frame_len]
            if np.sqrt(np.mean(chunk ** 2)) >= self.min_frame_rms:
                frames.append(chunk)
        return frames if frames else [wav[:self.frame_len]]

    def predict(self, wav):
        """16kHz float32 파형을 받아 (max_sim, is_anomaly, n_frames)를 반환한다."""
        wav = self.peak_normalize(np.asarray(wav, dtype=np.float32))
        frames = self._split_frames(wav)
        sims = []
        for frame in frames:
            _, embeddings, _ = self.yamnet(frame.astype(np.float32))
            emb = embeddings.numpy()[0]
            sims.append(
                float(cosine_similarity(emb.reshape(1, -1), self.centroid.reshape(1, -1))[0][0])
            )
        max_sim = max(sims)
        return max_sim, max_sim >= self.threshold, len(frames)
