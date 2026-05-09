"""MQTT 알림 발행 엔드포인트 — server → arduino 진동 명령."""

from fastapi import APIRouter

from ..service import alert_service

router = APIRouter()


@router.post("/send-alert")
async def send_alert(maker_id: str, direction: str):
    """server → arduino MQTT 진동 명령 발행.

    topic: crane/{maker_id}/vibration  또는  forklift/{maker_id}/vibration
    """
    return await alert_service.publish_alert(maker_id, direction)
