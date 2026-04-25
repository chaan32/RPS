"""
Fusion 알림 → MQTT 발행.

실제 아두이노 펌웨어(output/Arduino_vibration_output.ipynb) 기준:
  - 구독 토픽 : "crane/2/vibration" (단일)
  - 인식 payload: "left" / "back" / "right" / "all"
    (그 외 문자열은 "Unknown payload"로 무시됨)

사용:
  pred = FusionPrediction.from_model_output(risk_matrix)
  await publish_prediction(pred)                    # async (서버 안에서 사용)
  publish_prediction_sync(pred)                     # 동기 (realtime_camera.py 등)
"""

from __future__ import annotations

import os
import asyncio

from aiomqtt import Client, MqttError

from risk_output import (
    FusionPrediction,
    PairRisk,
    ThreatType,
    DANGER_THRESHOLD,
)


# ── 아두이노 펌웨어 기준 토픽/payload ──
# 펌웨어가 단일 토픽만 구독하므로 fork/dropzone 모두 동일 토픽으로 발행.
# direction(payload)으로 위협 종류를 구분: 지게차=back, 인양물=all
DEFAULT_TOPIC = "crane/2/vibration"

DEFAULT_THREAT_TO_TOPIC = {
    ThreatType.FORKLIFT: DEFAULT_TOPIC,
    ThreatType.DROPZONE: DEFAULT_TOPIC,
}
DEFAULT_THREAT_TO_DIRECTION = {
    ThreatType.FORKLIFT: "back",   # 지게차 충돌 → 후방 부저
    ThreatType.DROPZONE: "all",    # 인양물 진입 → 전체 부저
}


# ── 단발 발행: 토픽+payload 직접 지정 ──
async def publish_vibration(
    topic: str,
    direction: str,
    broker: str | None = None,
    timeout: float = 3.0,
) -> dict:
    """
    단일 진동 명령 발행.

    Args:
      topic     : 예) "crane/2/vibration"
      direction : "left" / "back" / "right" / "all" (펌웨어가 인식하는 4종)
      broker    : None이면 .env 의 MQTT_BROKER 사용
    Returns:
      {"status": "success"|"fail", ...}
    """
    if broker is None:
        broker = os.getenv("MQTT_BROKER", "127.0.0.1")
    try:
        async with Client(broker, timeout=timeout) as client:
            await client.publish(topic, payload=direction)
        return {"status": "success", "topic": topic, "message": direction}
    except MqttError as e:
        return {"status": "fail", "topic": topic, "error": str(e)}


# ── 단일 PairRisk 발행 (펌웨어 1-alert-per-3s 제약 대응) ──
async def publish_pair(
    pair: PairRisk,
    threat_to_topic: dict[ThreatType, str] | None = None,
    threat_to_direction: dict[ThreatType, str] | None = None,
    broker: str | None = None,
) -> dict:
    """단일 PairRisk → MQTT 발행 (1회)."""
    t2t = threat_to_topic or DEFAULT_THREAT_TO_TOPIC
    t2d = threat_to_direction or DEFAULT_THREAT_TO_DIRECTION
    topic = t2t.get(pair.threat_type)
    direction = t2d.get(pair.threat_type)
    if topic is None or direction is None:
        return {
            "status": "skip",
            "reason": f"no mapping for {pair.threat_type.value}",
            "pair": pair.json_key,
        }
    res = await publish_vibration(topic, direction, broker=broker)
    res["pair"] = pair.json_key
    res["prob"] = pair.prob
    return res


def publish_pair_sync(pair: PairRisk, **kwargs) -> dict:
    return asyncio.run(publish_pair(pair, **kwargs))


# ── FusionPrediction → 발행 ──
async def publish_prediction(
    prediction: FusionPrediction,
    threshold: float = DANGER_THRESHOLD,
    threat_to_topic: dict[ThreatType, str] | None = None,
    threat_to_direction: dict[ThreatType, str] | None = None,
    broker: str | None = None,
) -> list[dict]:
    """
    threshold 이상 trigger 된 PairRisk만 추려 MQTT 발행.
    같은 (topic, direction) 조합은 한 번만 발행한다 (forklift/dropzone이 같은 토픽일 때 중복 방지).
    """
    t2t = threat_to_topic or DEFAULT_THREAT_TO_TOPIC
    t2d = threat_to_direction or DEFAULT_THREAT_TO_DIRECTION

    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for p in prediction.triggered(threshold):
        topic = t2t.get(p.threat_type)
        direction = t2d.get(p.threat_type)
        if topic is None or direction is None:
            results.append({
                "status": "skip",
                "reason": f"no mapping for {p.threat_type.value}",
                "pair": p.json_key,
            })
            continue
        key = (topic, direction)
        if key in seen:
            continue
        seen.add(key)
        res = await publish_vibration(topic, direction, broker=broker)
        res["pair"] = p.json_key
        res["prob"] = p.prob
        results.append(res)
    return results


# ── 동기 래퍼 (realtime_camera.py 메인 루프용) ──
def publish_prediction_sync(
    prediction: FusionPrediction,
    threshold: float = DANGER_THRESHOLD,
    threat_to_topic: dict[ThreatType, str] | None = None,
    threat_to_direction: dict[ThreatType, str] | None = None,
    broker: str | None = None,
) -> list[dict]:
    """동기 컨텍스트(cv2 루프)에서 호출 가능."""
    return asyncio.run(publish_prediction(
        prediction,
        threshold=threshold,
        threat_to_topic=threat_to_topic,
        threat_to_direction=threat_to_direction,
        broker=broker,
    ))


# ── Sanity check ───────────────────────────────────────
def _sanity_check():
    """브로커 없이도 형식만 확인 (publish는 실제로 시도 → 실패해도 형식 검증 가능)."""
    import numpy as np

    fake_matrix = np.array([[0.92, 0.85]], dtype=np.float32)   # 둘 다 danger
    pred = FusionPrediction.from_model_output(fake_matrix)

    print("triggered pairs:")
    for p in pred.triggered():
        topic = DEFAULT_THREAT_TO_TOPIC[p.threat_type]
        direction = DEFAULT_THREAT_TO_DIRECTION[p.threat_type]
        print(f"  {p.threat_type.value:8s} prob={p.prob:.2f} "
              f"→ topic='{topic}'  payload='{direction}'")

    print("\n[publish] (실제 발행 시도)")
    results = publish_prediction_sync(pred, broker="127.0.0.1")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    _sanity_check()
