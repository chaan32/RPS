"""WAV 녹음 스크립트.

실행하면 카테고리 + 라벨을 선택하고,
Enter로 녹음 시작 → Enter로 녹음 종료 → 자동 저장을 반복한다.
q 입력 시 종료.

저장 경로: model/yamnet/dataset/{category}/{label}/{category}_{번호}.wav
포맷: 16kHz 모노 WAV (YAMNet 입력 규격)

카테고리:
  1) rope_stress  - 끊어지기 직전 소리 (삐걱, 장력)
  2) rope_release - 풀리는 소리

라벨:
  a) anomaly - 이상 소리
  n) normal  - 정상 소리
"""

import os
import threading

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def get_next_filename(folder_path, category, label):
    """다음 파일 번호를 자동으로 계산한다.

    anomaly면: rope_stress_001.wav
    normal이면: rope_stress_normal_001.wav
    """
    os.makedirs(folder_path, exist_ok=True)
    prefix = category if label == "anomaly" else f"{category}_normal"
    existing = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    return f"{prefix}_{len(existing)+1:03d}.wav"


def record_audio():
    """Enter로 시작, Enter로 종료하는 녹음."""
    frames = []
    stopped = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if not stopped.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    )

    with stream:
        input("  [녹음 중] Enter 누르면 종료...")
        stopped.set()

    if not frames:
        return None
    return np.concatenate(frames, axis=0)


def choose_category():
    print("카테고리를 선택하세요:")
    print("  1) rope_stress  (끊어지기 직전)")
    print("  2) rope_release (풀리는 소리)")
    print("  q) 종료")
    choice = input("선택: ").strip().lower()

    if choice == "q":
        return None
    elif choice == "1":
        return "rope_stress"
    elif choice == "2":
        return "rope_release"
    return "invalid"


def choose_label():
    print("라벨을 선택하세요:")
    print("  a) anomaly (이상 소리)")
    print("  n) normal  (정상 소리)")
    print("  b) 뒤로")
    choice = input("선택: ").strip().lower()

    if choice == "b":
        return None
    elif choice == "a":
        return "anomaly"
    elif choice == "n":
        return "normal"
    return "invalid"


def main():
    print(f"\n{'='*55}")
    print("  WAV 녹음기 (16kHz 모노)")
    print(f"{'='*55}")
    print(f"  저장 경로: {DATASET_DIR}")
    print(f"  사용법: 카테고리 → 라벨 → Enter로 녹음 시작/종료")
    print(f"{'='*55}\n")

    while True:
        category = choose_category()
        if category is None:
            print("종료합니다.")
            break
        if category == "invalid":
            print("잘못된 입력입니다.\n")
            continue

        label = choose_label()
        if label is None:
            print()
            continue
        if label == "invalid":
            print("잘못된 입력입니다.\n")
            continue

        folder = os.path.join(DATASET_DIR, category, label)
        filename = get_next_filename(folder, category, label)
        filepath = os.path.join(folder, filename)

        print(f"\n  카테고리: {category} / {label}")
        print(f"  파일명: {filename}")
        input("  Enter 누르면 녹음 시작...")

        audio = record_audio()

        if audio is None or len(audio) == 0:
            print("  녹음 실패!\n")
            continue

        duration = len(audio) / SAMPLE_RATE
        write(filepath, SAMPLE_RATE, audio)
        print(f"  저장 완료: {filepath} ({duration:.1f}초)\n")


if __name__ == "__main__":
    main()
