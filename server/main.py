from fastapi import FastAPI, Query
from aiomqtt import Client, MqttError
import os 
from .mqtt.mqtt_handler import MQTTHandler
import asyncio


app = FastAPI()

shared_pipeline = asyncio.Queue()

@app.get("/")
def read_root():
    return {"Hello": "FastAPI", "Status": "Running"}

@app.get("/hello/{name}")
def read_item(name : str):
    return f"HELLO + {name}"

@app.post("/send-vibration")
async def send_vibration(
    device_type: str = Query("crane", description="crane or forklift"),
    device_id: str = Query("01", description="아두이노 번호"),
    direction: str = Query("left", description="left or right or back")
):
    """
    if this API called, Server calls Ardoino
    """
    broker = os.getenv("MQTT_BROKER", "127.0.0.1")
    topic = f"{device_type}/{device_id}/{direction}" # topic (like crane/01/left)
    
    try:
        # off the connection after sending one message (like crane/01/left)
        async with Client(broker, timeout=3) as client:
            await client.publish(topic, payload=direction) # put the direction on payload 
            
        print(f"🚀 Succuess! [{topic}] -> {device_type}/{device_id}/{direction}")
        return {"status": "success", "topic": topic, "message": direction}
        
    except MqttError as e:
        print(f"❌ Sending Fail: {e}")
        return {"status": "fail", "error": "Fail at connection broker"}


# @app.on_event("startup")
# async def startup_event():
#     # generate MQTT handler instance and execute
#     mqtt_handler = MQTTHandler(shared_pipeline)
#     asyncio.create_task(mqtt_handler.run())


# @app.get("/status")
# async def get_latest_data():
#     if not shared_pipeline.empty():
#         data = await shared_pipeline.get()
#         return {"status": "success", "data": data}
#     return {"status": "empty", "message": "There isn't message"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("SERVER_PORT", 1122))
    uvicorn.run(app, host="0.0.0.0", port=port)