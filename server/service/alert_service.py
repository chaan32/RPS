"""MQTT 알림 발행 서비스 — server → arduino 진동 명령."""

import os

from aiomqtt import Client, MqttError


async def publish_alert(worker_id: str, direction: str) -> dict:
    """worker_id 에 맞는 토픽으로 direction 메시지를 MQTT 발행.

    topic 규칙:
      - worker/{id}/vibration
    """
    broker = os.getenv("MQTT_BROKER", "127.0.0.1")
    topic_template = os.getenv("WORKER_ALERT_TOPIC_TEMPLATE", "worker/{worker_id}/vibration")
    topic = topic_template.format(worker_id=worker_id)
    try:
        async with Client(broker, timeout=3) as client:
            await client.publish(topic, payload=direction)
        return {"status": "success", "topic": topic, "message": direction}
    except MqttError as e:
        return {"status": "fail", "error": str(e)}
