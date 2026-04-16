"""
오늘(KST) 날짜로 IncidentLog 더미 데이터를 생성하는 스크립트.

사용법:
    python -m server.scripts.seed_incident_logs
    python -m server.scripts.seed_incident_logs --count 30
    python -m server.scripts.seed_incident_logs --date 2026-04-15 --count 20
"""

import argparse
import asyncio
import random
from datetime import datetime, timedelta, timezone, date as date_cls

from sqlalchemy import select
from ..database import AsyncSessionLocal, IncidentLog, Maker


KST = timezone(timedelta(hours=9))

INCIDENT_TYPES = ["Warning", "Danger"]
STATUSES = ["success", "fail"]


def _random_dt_on(target: date_cls) -> datetime:
    start_kst = datetime(target.year, target.month, target.day, 0, 0, 0, tzinfo=KST)
    offset = timedelta(seconds=random.randint(0, 24 * 3600 - 1))
    dt_kst = start_kst + offset
    return dt_kst.astimezone(timezone.utc).replace(tzinfo=None)


async def seed(target: date_cls, count: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Maker.id))
        maker_ids = [row[0] for row in result.all()]
        if not maker_ids:
            raise RuntimeError("Makers가 없습니다. 서버를 한 번 실행해 시드해주세요.")

        logs = []
        for _ in range(count):
            logs.append(
                IncidentLog(
                    maker_id=random.choice(maker_ids),
                    incident_type=random.choices(INCIDENT_TYPES, weights=[7, 3])[0],
                    snapshot_path=f"https://s3.example.com/snapshots/{random.randint(1000, 9999)}.jpg",
                    status=random.choices(STATUSES, weights=[9, 1])[0],
                    created_at=_random_dt_on(target),
                )
            )

        session.add_all(logs)
        await session.commit()
        print(f"[OK] {target} (KST) 기준 IncidentLog {count}건 생성 완료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (KST). 생략 시 오늘")
    parser.add_argument("--count", type=int, default=15)
    args = parser.parse_args()

    target = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date else datetime.now(KST).date()
    )

    asyncio.run(seed(target, args.count))


if __name__ == "__main__":
    main()
