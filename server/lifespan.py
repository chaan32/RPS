"""FastAPI lifespan — 앱 시작/종료 시 실행할 것들 (DB init, MQTT, fusion subprocess)."""

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import select

from .database import AsyncSessionLocal, Base, Maker, engine
from .pipeline import MQTTHandler


async def mqtt_consumer(queue: asyncio.Queue):
    """MQTT 로 들어온 메시지를 큐에서 꺼내 처리 (현재는 로깅만)."""
    while True:
        data = await queue.get()
        print(f"🛠  Consumer got: {data}")
        # TODO: 필요 시 DB 저장 등 추가 처리


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 (yield 전) / 종료 시 (yield 후) 실행할 작업 정의."""
    # 1) DB 테이블 자동 생성 (없을 때만)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2) Maker 1~5 시드 (없을 때만)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Maker))
        if not result.scalars().first():
            session.add_all([Maker(id=i, count=0) for i in range(1, 6)])
            await session.commit()

    # 3) Fusion 추론 subprocess 기동
    # model.fusion.runtime.realtime_camera 는 캘리브레이션 자동 보장 + YOLO + ArUco +
    # Fusion 모델 추론 + MQTT/DB 발행까지 한 번에 처리.
    # cwd 를 PROJECT_ROOT 로 두어 sys.path 자동 포함.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cam_proc = subprocess.Popen(
        [sys.executable, "-m", "model.fusion.runtime.realtime_camera", "--no-prompt"],
        cwd=project_root,
    )

    # 4) MQTT 파이프라인 (백그라운드 producer/consumer)
    queue: asyncio.Queue = asyncio.Queue()
    handler = MQTTHandler(queue)
    producer_task = asyncio.create_task(handler.run())
    consumer_task = asyncio.create_task(mqtt_consumer(queue))

    try:
        yield
    finally:
        # 종료 시 정리
        producer_task.cancel()
        consumer_task.cancel()
        cam_proc.terminate()
