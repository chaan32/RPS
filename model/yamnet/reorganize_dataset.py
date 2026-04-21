"""데이터셋 폴더 재구성 스크립트.

기존 anomaly/ 폴더의 파일들을 분류하여 새 폴더 구조로 이동한다.
mp3 파일은 16kHz 모노 WAV로 자동 변환한다.

새 구조:
  dataset/
  ├── rope_stress/anomaly/   ← 끊어지기 직전 (삐걱, 장력, 비틀림)
  ├── rope_stress/normal/
  ├── rope_release/anomaly/  ← 풀리는 소리 (빠르게 풀림)
  └── rope_release/normal/

이름 규칙: rope_stress_001.wav, rope_release_001.wav
"""

import os
import shutil
import glob

from pydub import AudioSegment

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OLD_ANOMALY = os.path.join(DATASET_DIR, "anomaly")
OLD_NORMAL = os.path.join(DATASET_DIR, "normal")

# ── 분류 규칙 ──────────────────────────────────────────────────────────
# 파일 이름에 이 키워드가 포함되면 해당 카테고리로 분류
RELEASE_KEYWORDS = ["fast"]  # fast = 빠르게 풀리는 소리
# 나머지는 전부 rope_stress (끊어지기 직전)


def ensure_dirs():
    """새 폴더 구조 생성."""
    for category in ["rope_stress", "rope_release"]:
        for label in ["anomaly", "normal"]:
            os.makedirs(os.path.join(DATASET_DIR, category, label), exist_ok=True)


def to_wav_16k(src_path, dst_path):
    """오디오 파일을 16kHz 모노 WAV로 변환/복사."""
    if src_path.endswith(".wav"):
        # WAV도 16kHz 모노로 통일
        audio = AudioSegment.from_file(src_path)
    else:
        audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(dst_path, format="wav")


def classify_file(filename):
    """파일 이름으로 카테고리를 분류한다."""
    lower = filename.lower()
    for kw in RELEASE_KEYWORDS:
        if kw in lower:
            return "rope_release"
    return "rope_stress"


def main():
    ensure_dirs()

    print(f"\n{'='*55}")
    print("  데이터셋 재구성")
    print(f"{'='*55}\n")

    counters = {
        "rope_stress": {"anomaly": 0, "normal": 0},
        "rope_release": {"anomaly": 0, "normal": 0},
    }

    # 1. 기존 anomaly 폴더 파일 분류
    if os.path.exists(OLD_ANOMALY):
        files = sorted(os.listdir(OLD_ANOMALY))
        print(f"[기존 anomaly/] {len(files)}개 파일 처리 중...\n")

        for fname in files:
            src = os.path.join(OLD_ANOMALY, fname)
            if not os.path.isfile(src):
                continue

            category = classify_file(fname)
            label = "anomaly"
            counters[category][label] += 1
            idx = counters[category][label]
            new_name = f"{category}_{idx:03d}.wav"
            dst = os.path.join(DATASET_DIR, category, label, new_name)

            to_wav_16k(src, dst)
            print(f"  {fname}")
            print(f"    → {category}/{label}/{new_name}")

    # 2. 기존 normal 폴더 파일 → rope_stress/normal로 이동
    if os.path.exists(OLD_NORMAL):
        files = sorted(os.listdir(OLD_NORMAL))
        print(f"\n[기존 normal/] {len(files)}개 파일 처리 중...\n")

        for fname in files:
            src = os.path.join(OLD_NORMAL, fname)
            if not os.path.isfile(src):
                continue

            category = "rope_stress"  # normal은 기본적으로 rope_stress에
            label = "normal"
            counters[category][label] += 1
            idx = counters[category][label]
            new_name = f"{category}_normal_{idx:03d}.wav"
            dst = os.path.join(DATASET_DIR, category, label, new_name)

            to_wav_16k(src, dst)
            print(f"  {fname}")
            print(f"    → {category}/{label}/{new_name}")

    # 결과 출력
    print(f"\n{'='*55}")
    print("  결과")
    print(f"{'='*55}")
    for cat, labels in counters.items():
        for label, count in labels.items():
            print(f"  {cat}/{label}: {count}개")
    print(f"{'='*55}")

    # 기존 폴더 삭제 확인
    print(f"\n기존 anomaly/, normal/ 폴더를 삭제할까요?")
    choice = input("삭제하려면 'y' 입력: ").strip()
    if choice.lower() == "y":
        if os.path.exists(OLD_ANOMALY):
            shutil.rmtree(OLD_ANOMALY)
        if os.path.exists(OLD_NORMAL):
            shutil.rmtree(OLD_NORMAL)
        print("기존 폴더 삭제 완료!")
    else:
        print("기존 폴더를 유지합니다.")


if __name__ == "__main__":
    main()
