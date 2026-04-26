import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# USB 마운트 경로 (macOS: /Volumes/USB이름, Linux: /media/usb 등)
USB_BASE_PATH = os.getenv("USB_STORE_PATH", "/Volumes/USB")

# USB가 마운트되어 있지 않을 때 떨어뜨릴 로컬 폴백 디렉토리.
# 기본값: 프로젝트 루트의 snapshots/ 폴더.
# 이게 없으면 USB 미연결 시 save_file 이 PermissionError 등으로 깨져서
# /incident-logs/with-snapshot 이 500 → DB 행도 안 들어감.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_FALLBACK_PATH = os.getenv(
    "LOCAL_SNAPSHOT_PATH", str(_PROJECT_ROOT / "snapshots"),
)


def _resolve_base() -> Path:
    """USB가 마운트되어 있으면 USB, 아니면 로컬 폴백 경로를 돌려준다."""
    base = Path(USB_BASE_PATH)
    if base.exists():
        return base
    return Path(LOCAL_FALLBACK_PATH)


def save_file(file_bytes: bytes, key: str, content_type: str = "image/jpeg") -> str:
    """
    S3의 upload_file과 같은 역할.
    USB가 꽂혀 있으면 USB에, 없으면 LOCAL_FALLBACK_PATH 에 바이트를 저장하고
    파일 경로를 반환한다.

    key: 저장 경로 (예: 'snapshots/2026-04-18/abc.jpg')
    반환: 저장된 파일의 절대 경로
    """
    save_path = _resolve_base() / key

    # 상위 폴더가 없으면 자동 생성 (snapshots/2026-04-18/ 등)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_path.write_bytes(file_bytes)

    return str(save_path)


def is_usb_connected() -> bool:
    """USB가 마운트되어 있는지 확인"""
    return Path(USB_BASE_PATH).exists()
