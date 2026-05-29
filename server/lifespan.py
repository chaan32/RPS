"""FastAPI lifespan — 앱 시작/종료 시 실행할 것들 (DB init, MQTT, fusion subprocess)."""

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import select, text

from .database import AsyncSessionLocal, Base, Worker, engine
from .jobs.redis_queue import redis_worker_loop
from .pipeline import MQTTHandler


async def mqtt_consumer(queue: asyncio.Queue):
    """MQTT 로 들어온 메시지를 큐에서 꺼내 처리 (현재는 로깅만)."""
    while True:
        data = await queue.get()
        print(f"🛠  Consumer got: {data}")
        # TODO: 필요 시 DB 저장 등 추가 처리


async def _migrate_worker_schema(conn) -> None:
    """Rename old maker schema to worker schema and keep only worker ids 1,2."""
    await conn.execute(text("""
        DO $$
        BEGIN
            IF to_regclass('public.workers') IS NULL
               AND to_regclass('public.makers') IS NOT NULL THEN
                ALTER TABLE makers RENAME TO workers;
            END IF;

            IF to_regclass('public.incident_logs') IS NOT NULL
               AND EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'incident_logs'
                      AND column_name = 'maker_id'
               )
               AND NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'incident_logs'
                      AND column_name = 'worker_id'
               ) THEN
                ALTER TABLE incident_logs RENAME COLUMN maker_id TO worker_id;
            END IF;
        END $$;
    """))


async def _normalize_workers(conn) -> None:
    """Seed workers 1/2 and map legacy incident rows to those workers."""
    await conn.execute(text("""
        INSERT INTO workers (id, count)
        VALUES (1, 0), (2, 0)
        ON CONFLICT (id) DO NOTHING
    """))
    await conn.execute(text("""
        DO $$
        BEGIN
            IF to_regclass('public.incident_logs') IS NOT NULL THEN
                UPDATE incident_logs
                SET worker_id = CASE WHEN worker_id = 1 THEN 1 ELSE 2 END
                WHERE worker_id NOT IN (1, 2);

                UPDATE workers
                SET count = sub.total
                FROM (
                    SELECT worker_id, COUNT(*)::int AS total
                    FROM incident_logs
                    GROUP BY worker_id
                ) AS sub
                WHERE workers.id = sub.worker_id
                  AND workers.id IN (1, 2);
            END IF;
        END $$;
    """))
    await conn.execute(text("DELETE FROM workers WHERE id NOT IN (1, 2)"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 (yield 전) / 종료 시 (yield 후) 실행할 작업 정의."""
    # 1) DB 테이블 자동 생성 (없을 때만)
    async with engine.begin() as conn:
        await _migrate_worker_schema(conn)
        await conn.run_sync(Base.metadata.create_all)
        await _normalize_workers(conn)
        # create_all()은 이미 존재하는 컬럼 타입을 바꾸지 않는다.
        # 예전 DB에서는 reports.contents가 varchar(3000)이라 LLM HTML 저장이 실패하므로
        # TEXT로 확장해 긴 리포트를 안전하게 저장한다.
        await conn.execute(
            text("ALTER TABLE reports ALTER COLUMN contents TYPE TEXT")
        )
        # 수동 seed/복원 데이터가 있으면 serial sequence가 max(id)보다 낮아질 수 있다.
        # 이 경우 새 리포트 INSERT가 duplicate key로 실패하므로 시작 시 동기화한다.
        await conn.execute(text("""
            SELECT setval(
                pg_get_serial_sequence('reports', 'id'),
                COALESCE((SELECT MAX(id) FROM reports), 1),
                (SELECT MAX(id) IS NOT NULL FROM reports)
            )
        """))

    # 2) Worker 1~2 시드 (없을 때만)
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Worker))
        if not result.scalars().first():
            session.add_all([Worker(id=i, count=0) for i in range(1, 3)])
            await session.commit()

    # 3) Fusion 추론 subprocess 기동
    # model.fusion.runtime.realtime_camera 는 캘리브레이션 자동 보장 + YOLO + ArUco +
    # Fusion 모델 추론 + MQTT/DB 발행까지 한 번에 처리.
    # cwd 를 PROJECT_ROOT 로 두어 sys.path 자동 포함.
    #
    # RTSP bridge / 성능평가처럼 realtime process 를 명시적으로 따로 실행할 때는
    # DISABLE_FUSION_SUBPROCESS=1 로 서버 API/DB/MQTT 만 켠다.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cam_proc = None
    if os.getenv("DISABLE_FUSION_SUBPROCESS", "").lower() not in ("1", "true", "yes"):
        cam_proc = subprocess.Popen(
            [sys.executable, "-m", "model.fusion.runtime.realtime_camera", "--no-prompt"],
            cwd=project_root,
        )

    # 4) MQTT 파이프라인 (백그라운드 producer/consumer)
    queue: asyncio.Queue = asyncio.Queue()
    handler = MQTTHandler(queue)
    producer_task = asyncio.create_task(handler.run())
    consumer_task = asyncio.create_task(mqtt_consumer(queue))
    redis_job_task = None
    if os.getenv("DISABLE_REDIS_JOBS", "").lower() not in ("1", "true", "yes"):
        redis_job_task = asyncio.create_task(redis_worker_loop())

    try:
        yield
    finally:
        # 종료 시 정리
        producer_task.cancel()
        consumer_task.cancel()
        if redis_job_task is not None:
            redis_job_task.cancel()
        if cam_proc is not None:
            cam_proc.terminate()
