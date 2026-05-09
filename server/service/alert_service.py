"""MQTT 알림 발행 서비스 — server → arduino 진동 명령."""

import os

from aiomqtt import Client, MqttError


async def publish_alert(maker_id: str, direction: str) -> dict:
    """maker_id 에 맞는 토픽으로 direction 메시지를 MQTT 발행.

    topic 규칙:
      - maker_id == '4' or '5' → forklift/{id}/vibration
      - 그 외                  → crane/{id}/vibration
    """
    broker = os.getenv("MQTT_BROKER", "127.0.0.1")
    topic = f"crane/{maker_id}/vibration"
    if maker_id == "5" or maker_id == "4":
        topic = f"forklift/{maker_id}/vibration"
    try:
        async with Client(broker, timeout=3) as client:
            await client.publish(topic, payload=direction)
        return {"status": "success", "topic": topic, "message": direction}
    except MqttError as e:
        return {"status": "fail", "error": str(e)}
