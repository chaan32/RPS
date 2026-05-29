"""MQTT 알림 발행 엔드포인트 — server → arduino 진동 명령."""

from fastapi import APIRouter

from ..service import alert_service

router = APIRouter()


@router.post("/send-alert")
async def send_alert(direction: str, worker_id: str | None = None, maker_id: str | None = None):
    """server → arduino MQTT 진동 명령 발행.

    topic: worker/{worker_id}/vibration
    """
    resolved_worker_id = worker_id or maker_id or "1"
    return await alert_service.publish_alert(resolved_worker_id, direction)
