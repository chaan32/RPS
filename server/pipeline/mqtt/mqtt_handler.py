"""MQTT broker 메시지를 FastAPI 내부 asyncio 큐로 전달한다.

실제 위험 이벤트 저장은 Fusion runtime의 DB 로거가 담당한다. 이 핸들러는
외부 장치 메시지를 서버 프로세스 안으로 받아오는 producer 역할만 수행한다.
"""

import os
import asyncio
from aiomqtt import Client, MqttError
from dotenv import load_dotenv

load_dotenv(override=True)


class MQTTHandler:
    """환경 변수에 등록된 MQTT topic을 구독하고 수신 메시지를 큐에 넣는다."""

    def __init__(self, message_queue: asyncio.Queue):
        self.broker = os.getenv("MQTT_BROKER")
        self.message_queue = message_queue
        self.topics = [
            value
            for key, value in os.environ.items()
            if key.startswith("MQTT_") and key != "MQTT_BROKER"
        ]

    async def run(self):
        """MQTT 연결이 끊기면 30초 뒤 재연결을 시도한다."""
        print(f"[mqtt] connecting broker={self.broker}")
        while True:
            try:
                # 브로커 장애 시 무한 대기하지 않도록 연결 타임아웃을 둔다.
                async with Client(self.broker, timeout=5) as client:
                    for topic in self.topics:
                        await client.subscribe(topic)

                    async for message in client.messages:
                        payload = message.payload.decode()
                        topic = message.topic.value

                        data = {"topic": topic, "message": payload}
                        await self.message_queue.put(data)
                        print(f"[mqtt] queue in: {data}")

            except MqttError as e:
                print(f"[mqtt] broker connection failed; retry in 30s ({e})")
                await asyncio.sleep(30)
            except Exception as e:
                print(f"[mqtt] unexpected error; retry in 30s ({e})")
                await asyncio.sleep(30)
