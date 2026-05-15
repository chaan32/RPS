"""Unity 가 export 한 JSON 시나리오들을 Scenario 객체 리스트로 로드.

Unity 측 (simulation/Assets/Scripts/) 의 BatchScenarioGenerator 가
Assets/Output/*.json 으로 떨어뜨린 파일들을 학습 파이프라인에서 사용할 수 있는
Scenario 형식으로 변환한다.

JSON 형식 (Unity 측 ScenarioData 와 일치):
    {
      "name": "unity_safe_001",
      "scenario_type": "safe" | "fork" | "dz",
      "rate_hz": 5.0,
      "duration_sec": 20.0,
      "frames": [
        {
          "t": 0.0,
          "worker_x": -0.3, "worker_y": 1.5,
          "forklift_present": true,
          "forklift_x": -2.0, "forklift_y": 0.0,
          "dropzone_x": -1.5, "dropzone_y": 2.0,
          "crane_active": 0,
          "audio_score": 0.05
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scenario_generator import Scenario


def load_unity_scenarios(json_dir: str | Path) -> list[Scenario]:
    """디렉터리 안 모든 *.json → Scenario 객체 리스트.

    Args:
        json_dir: Unity Assets/Output 디렉터리 또는 사본 위치.

    Returns:
        list[Scenario]: 변환된 시나리오. 파싱 실패한 파일은 skip + 경고 출력.
    """
    json_dir = Path(json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"디렉터리 없음: {json_dir}")

    scenarios: list[Scenario] = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            scenarios.append(_load_one(path))
        except Exception as e:
            print(f"[unity-loader skip] {path.name}: {e}")

    print(f"[unity-loader] {len(scenarios)} scenarios loaded from {json_dir}")
    return scenarios


def _load_one(path: Path) -> Scenario:
    """JSON 파일 1개 → Scenario."""
    with open(path) as f:
        data = json.load(f)

    frames = data["frames"]
    n = len(frames)

    worker1 = np.zeros((n, 2), dtype=np.float32)
    forklift = np.zeros((n, 2), dtype=np.float32)
    crane_state = np.zeros(n, dtype=np.int32)
    audio = np.zeros(n, dtype=np.float32)

    for i, fr in enumerate(frames):
        worker1[i] = [fr["worker_x"], fr["worker_y"]]

        # forklift_present == False 면 NaN sentinel (graph_input 이 자동 처리)
        if fr.get("forklift_present", True):
            forklift[i] = [fr["forklift_x"], fr["forklift_y"]]
        else:
            forklift[i] = [np.nan, np.nan]

        crane_state[i] = int(fr.get("crane_active", 0))
        audio[i] = float(fr.get("audio_score", 0.0))

    # Scenario 의 name 은 학습 split 에서 카테고리 판별에 쓰임
    # (_scenario_category 가 "_safe_", "_fork_", "_dz_" 토큰을 봄)
    return Scenario(
        name=data["name"],
        forklift=forklift,
        worker1=worker1,
        crane_state=crane_state,
        audio=audio,
        labels=None,   # Python 의 compute_pair_labels 가 자동 계산
    )


# ── 단독 실행 시 sanity check ─────────────────────────────────
def _sanity_check():
    """Assets/Output 의 JSON 들 로드 + 기본 통계 출력."""
    import sys
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    default_dir = project_root / "simulation" / "Assets" / "Output"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    scenarios = load_unity_scenarios(target)
    if not scenarios:
        print("[sanity] 시나리오 0개 — Unity 에서 'Generate Scenarios' 먼저 실행하세요.")
        return

    by_type: dict[str, int] = {}
    for s in scenarios:
        for tag in ("safe", "fork", "dz"):
            if f"_{tag}_" in s.name:
                by_type[tag] = by_type.get(tag, 0) + 1
                break

    print(f"\n=== Unity 시나리오 통계 ===")
    print(f"  total      : {len(scenarios)}")
    for tag, cnt in by_type.items():
        print(f"  {tag:8s}  : {cnt}")

    # 첫 시나리오 형태 점검
    s0 = scenarios[0]
    print(f"\n=== 첫 시나리오 ({s0.name}) ===")
    print(f"  worker1     shape={s0.worker1.shape}, dtype={s0.worker1.dtype}")
    print(f"  forklift    shape={s0.forklift.shape}, NaN frames={int(np.isnan(s0.forklift).any(axis=1).sum())}")
    print(f"  crane_state shape={s0.crane_state.shape}, sum={s0.crane_state.sum()}")
    print(f"  audio       shape={s0.audio.shape}, range=[{s0.audio.min():.2f}, {s0.audio.max():.2f}]")


if __name__ == "__main__":
    _sanity_check()
