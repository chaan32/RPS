# Parallel Benchmark Matrix - pose skip 2

## Purpose

This run tests pose inference scheduling. The pipeline keeps `POSE_IMGSZ=640` for worker reliability, keeps `CUSTOM_IMGSZ=512` for the custom forklift model, and runs YOLO pose every 2 frames.

- Pose model input: `POSE_IMGSZ=640`
- Custom model input: `CUSTOM_IMGSZ=512`
- Pose schedule: `POSE_EVERY_N_FRAMES=2`
- Scenarios: `scenario_01_user_current`, `scenario_02_swapped_positions`, `scenario_03_opposite_worker`
- Modes: `serial`, `camera_parallel`, `model_parallel`
- Repeats: 5 per mode/scenario, 45 total runs
- Runtime duration per run: 35s

## Overall Results

| Mode | Runs | Avg FPS | FPS Range | Avg Loop | Pose Skipped | Worker Rate | Forklift Rate | Prediction Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| serial | 15 | 5.443 | 4.755-5.846 | 185.259 ms | 45.20% | 1.000 | 0.980 | 0.978 |
| camera_parallel | 15 | 8.982 | 7.689-10.089 | 112.098 ms | 45.49% | 1.000 | 0.985 | 0.987 |
| model_parallel | 15 | 10.148 | 9.035-11.065 | 98.966 ms | 45.76% | 1.000 | 0.985 | 0.989 |

## Scenario Results

| Mode | Scenario | Runs | Avg FPS | Avg Loop | Pose Skipped | Worker Rate | Forklift Rate | Prediction Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| serial | scenario_01_user_current | 5 | 5.679 | 176.767 ms | 0.4828 | 0.999 | 1.000 | 0.979 |
| serial | scenario_02_swapped_positions | 5 | 4.876 | 205.639 ms | 0.3747 | 1.000 | 0.939 | 0.977 |
| serial | scenario_03_opposite_worker | 5 | 5.774 | 173.371 ms | 0.4985 | 1.000 | 1.000 | 0.980 |
| camera_parallel | scenario_01_user_current | 5 | 9.423 | 106.058 ms | 0.4839 | 1.000 | 1.000 | 0.988 |
| camera_parallel | scenario_02_swapped_positions | 5 | 7.962 | 125.529 ms | 0.3820 | 1.000 | 0.956 | 0.986 |
| camera_parallel | scenario_03_opposite_worker | 5 | 9.562 | 104.706 ms | 0.4988 | 1.000 | 1.000 | 0.988 |
| model_parallel | scenario_01_user_current | 5 | 10.664 | 93.649 ms | 0.4858 | 1.000 | 1.000 | 0.989 |
| model_parallel | scenario_02_swapped_positions | 5 | 9.135 | 109.333 ms | 0.3883 | 1.000 | 0.956 | 0.988 |
| model_parallel | scenario_03_opposite_worker | 5 | 10.645 | 93.917 ms | 0.4987 | 1.000 | 1.000 | 0.989 |

## Comparison To pose640/custom512 Without Pose Skip

Previous benchmark: `metrics/parallel_matrix_20260520_custom512`

| Mode | Previous FPS | Pose Skip FPS | Change | Previous Loop | Pose Skip Loop | Loop Change |
|---|---:|---:|---:|---:|---:|---:|
| serial | 3.489 | 5.443 | +56.0% | 289.581 ms | 185.259 ms | -36.0% |
| camera_parallel | 6.389 | 8.982 | +40.6% | 156.524 ms | 112.098 ms | -28.4% |
| model_parallel | 7.364 | 10.148 | +37.8% | 135.510 ms | 98.966 ms | -27.7% |

## Interpretation

Pose scheduling is the strongest optimization so far.

- Worker detection stayed stable at 100% across the 45-run benchmark.
- Forklift detection did not regress overall. It remains weakest in `scenario_02_swapped_positions`, where visibility/model confidence was already the limiting point.
- The final goal of approaching 10 FPS is now reached in `model_parallel`: average 10.148 FPS.
- `scenario_02_swapped_positions` skips less pose work because one camera often has no worker cache, so that camera keeps running pose. This is expected and avoids hallucinating a worker where no recent worker was seen.

## Current Recommended Runtime Setting

Use this combination for the current Unity scenario set:

```env
POSE_MODEL_PATH=model/yolo/yolo11s-pose.pt
WORKER_WORLD_BOUNDS=none
POSE_IMGSZ=640
CUSTOM_IMGSZ=512
POSE_EVERY_N_FRAMES=2
```

Keep `model_parallel` as the default extraction mode for performance benchmarks.
