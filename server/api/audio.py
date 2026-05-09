"""오디오 score 노출 — fusion subprocess 가 폴링."""

from fastapi import APIRouter

from input.audio import get_latest_score

router = APIRouter()


@router.get("/audio/score")
def audio_score():
    """ESP32 yamnet 최근 결과. realtime_camera.py 가 fusion 입력으로 사용."""
    return get_latest_score()
