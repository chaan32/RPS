"""데이터셋 최종 정리 스크립트.

구조를 다음과 같이 이진 분류용으로 정리:

  dataset/
  ├── anomaly/    ← rope_stress/anomaly + rope_release/anomaly 통합
  │   ├── rope_stress_001.wav ~ rope_stress_041.wav
  │   └── rope_release_001.wav ~ rope_release_050.wav
  └── normal/     ← winch 소음 (m4a는 wav로 자동 변환)
      ├── winch_001.wav ~ winch_051.wav

기존 rope_stress/, rope_release/, winch/ 폴더는 삭제.
엉뚱한 파일(pill, curiousOfTodaysLunch 등)도 제거.
"""

import os
import shutil
import glob

from pydub import AudioSegment

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def ensure_dirs():
    for label in ["anomaly", "normal"]:
        os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)


def to_wav_16k(src_path, dst_path):
    """오디오 파일을 16kHz 모노 WAV로 변환."""
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(dst_path, format="wav")


def clear_dir(dir_path):
    """폴더 안 파일 전부 삭제 (폴더 구조 유지)."""
    if not os.path.exists(dir_path):
        return
    for f in os.listdir(dir_path):
        fpath = os.path.join(dir_path, f)
        if os.path.isfile(fpath):
            os.remove(fpath)


def main():
    ensure_dirs()
    print(f"\n{'='*55}")
    print("  데이터셋 최종 정리 (anomaly vs normal)")
    print(f"{'='*55}\n")

    # 기존 anomaly/, normal/ 폴더 비우기 (혹시 남아있을 수 있는 파일 제거)
    clear_dir(os.path.join(DATASET_DIR, "anomaly"))
    clear_dir(os.path.join(DATASET_DIR, "normal"))

    # 1. rope_stress/anomaly → anomaly/rope_stress_XXX.wav
    src_stress = os.path.join(DATASET_DIR, "rope_stress", "anomaly")
    stress_files = sorted([f for f in os.listdir(src_stress) if f.endswith('.wav')])
    print(f"[rope_stress] {len(stress_files)}개 파일 이동")
    for i, fname in enumerate(stress_files, 1):
        src = os.path.join(src_stress, fname)
        dst = os.path.join(DATASET_DIR, "anomaly", f"rope_stress_{i:03d}.wav")
        to_wav_16k(src, dst)
    print(f"  → anomaly/rope_stress_001.wav ~ rope_stress_{len(stress_files):03d}.wav")

    # 2. rope_release/anomaly → anomaly/rope_release_XXX.wav
    src_release = os.path.join(DATASET_DIR, "rope_release", "anomaly")
    release_files = sorted([f for f in os.listdir(src_release) if f.endswith('.wav')])
    print(f"\n[rope_release] {len(release_files)}개 파일 이동")
    for i, fname in enumerate(release_files, 1):
        src = os.path.join(src_release, fname)
        dst = os.path.join(DATASET_DIR, "anomaly", f"rope_release_{i:03d}.wav")
        to_wav_16k(src, dst)
    print(f"  → anomaly/rope_release_001.wav ~ rope_release_{len(release_files):03d}.wav")

    # 3. winch/*.m4a → normal/winch_XXX.wav
    src_winch = os.path.join(DATASET_DIR, "winch")
    winch_files = sorted([
        f for f in os.listdir(src_winch)
        if f.lower().endswith(('.m4a', '.mp3', '.wav', '.ogg', '.flac', '.aac'))
    ])
    print(f"\n[winch] {len(winch_files)}개 파일 변환 + 이동")
    for i, fname in enumerate(winch_files, 1):
        src = os.path.join(src_winch, fname)
        dst = os.path.join(DATASET_DIR, "normal", f"winch_{i:03d}.wav")
        to_wav_16k(src, dst)
    print(f"  → normal/winch_001.wav ~ winch_{len(winch_files):03d}.wav")

    # 4. 기존 폴더 + 엉뚱한 파일 삭제
    print(f"\n[정리]")
    for folder in ["rope_stress", "rope_release", "winch"]:
        fpath = os.path.join(DATASET_DIR, folder)
        if os.path.exists(fpath):
            shutil.rmtree(fpath)
            print(f"  삭제: {folder}/")

    # 엉뚱한 파일 삭제
    for f in os.listdir(DATASET_DIR):
        fpath = os.path.join(DATASET_DIR, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"  삭제: {f}")

    # 결과 요약
    print(f"\n{'='*55}")
    print(f"  최종 결과")
    print(f"{'='*55}")
    for label in ["anomaly", "normal"]:
        ldir = os.path.join(DATASET_DIR, label)
        files = sorted(os.listdir(ldir))
        print(f"  {label}/ ({len(files)}개)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
