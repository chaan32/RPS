import os
import asyncio
from aiomqtt import Client, MqttError
from dotenv import load_dotenv

load_dotenv(override=True)

class MQTTHandler:
    def __init__(self, message_queue: asyncio.Queue):
        self.broker = os.getenv("MQTT_BROKER")
        self.message_queue = message_queue
        # bring all topics from .env
        self.topics = [v for k, v in os.environ.items() if k.startswith("MQTT_") and k != "MQTT_BROKER"]

    async def run(self):
        print(f"Connecting to MQTT Broker at {self.broker}...")
        while True:
            try:
                # 타임아웃을 주면 브로커가 죽었을 때 무한 대기하는 걸 막아줍니다.
                async with Client(self.broker, timeout=5) as client:
                    # subscribe all topics 
                    for topic in self.topics:
                        await client.subscribe(topic)
            

                    # 🚨 async with 줄을 삭제하고 바로 async for 로 진입합니다!
                    async for message in client.messages:
                        payload = message.payload.decode()
                        topic = message.topic.value
                        
                        # Topic and payload 
                        data = {"topic": topic, "message": payload}
                        await self.message_queue.put(data)
                        print(f"📩 Queue In: {data}")
                        
            except MqttError as e:
                print(f"❌ MQTT Fail: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"⚠️ Unexpected Error: {e}")
                await asyncio.sleep(5)