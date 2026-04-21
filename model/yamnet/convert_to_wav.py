"""오디오 파일 → WAV 일괄 변환 스크립트.

dataset/anomaly (또는 normal) 폴더 안의 mp3, m4a, ogg, flac, aac, wma 파일을
16kHz 모노 WAV로 변환하고 원본은 삭제한다.

실행: python model/yamnet/convert_to_wav.py
"""

import os
import glob

from pydub import AudioSegment

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TARGET_EXTENSIONS = ("*.mp3", "*.m4a", "*.ogg", "*.flac", "*.aac", "*.wma")


def convert_folder(folder_path):
    """폴더 안의 비-WAV 오디오 파일을 전부 WAV로 변환한다."""
    converted = 0
    for ext in TARGET_EXTENSIONS:
        for src_path in glob.glob(os.path.join(folder_path, ext)):
            wav_path = os.path.splitext(src_path)[0] + ".wav"
            try:
                audio = AudioSegment.from_file(src_path)
                # 16kHz 모노로 변환 (YAMNet 규격)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(wav_path, format="wav")
                os.remove(src_path)
                converted += 1
                print(f"  {os.path.basename(src_path)} -> {os.path.basename(wav_path)}")
            except Exception as e:
                print(f"  [FAIL] {os.path.basename(src_path)}: {e}")
    return converted


def main():
    print(f"\n{'='*50}")
    print("  오디오 → WAV 변환기 (16kHz 모노)")
    print(f"{'='*50}")
    print(f"  대상 폴더: {DATASET_DIR}")
    print(f"  변환 대상: {', '.join(e.replace('*', '') for e in TARGET_EXTENSIONS)}")
    print(f"{'='*50}\n")

    total = 0
    for category in os.listdir(DATASET_DIR):
        cat_dir = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        print(f"[{category}]")
        count = convert_folder(cat_dir)
        total += count
        if count == 0:
            print("  변환할 파일 없음")
        print()

    print(f"총 {total}개 파일 변환 완료!")


if __name__ == "__main__":
    main()
