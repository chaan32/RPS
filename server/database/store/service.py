import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# USB 마운트 경로 (macOS: /Volumes/USB이름, Linux: /media/usb 등)
USB_BASE_PATH = os.getenv("USB_STORE_PATH", "/Volumes/USB")


def save_file(file_bytes: bytes, key: str, content_type: str = "image/jpeg") -> str:
    """
    S3의 upload_file과 같은 역할.
    바이트 데이터를 USB 드라이브에 저장하고 파일 경로를 반환한다.

    key: 저장 경로 (예: 'snapshots/2026-04-18/abc.jpg')
    반환: 저장된 파일의 절대 경로
    """
    save_path = Path(USB_BASE_PATH) / key

    # 상위 폴더가 없으면 자동 생성 (snapshots/2026-04-18/ 등)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_path.write_bytes(file_bytes)

    return str(save_path)


def is_usb_connected() -> bool:
    """USB가 마운트되어 있는지 확인"""
    return Path(USB_BASE_PATH).exists()
