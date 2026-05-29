# Fusion V2

Fusion V2 is the deep-learning risk prediction track.  The production V1
runtime remains in `model/fusion` and is not replaced by this package.

## Purpose

V1 decides `SAFE`, `WARNING`, and `DANGER` from BEV/world coordinates with
explicit rules: distance, TTC, forklift front-hazard point, and DropZone
radius.  V2 keeps the same perception-to-coordinate input format, but learns
the final risk decision from recent coordinate sequences.

```text
YOLO/Pose -> Homography BEV coordinates -> coordinate window -> GRU risk model
```

## Independence Level

There are two supported label modes.

| Mode | Label Source | Meaning |
| --- | --- | --- |
| `teacher` | V1 risk columns from `fusion_risk.csv` | Bootstraps a model that imitates V1 decisions. |
| `geometry_future` | Absolute coordinates only | Trains V2 from distance/future overlap rules without using V1 risk scores. |

The latest V2 run uses `geometry_future`.  It does **not** use `forklift_risk`,
`dropzone_risk`, `early_level`, or `dropzone_forced` as labels.  It still uses
the coordinate traces produced by the existing perception pipeline.  A fully
perception-independent dataset would require Unity ground-truth object
positions exported directly from the simulator.

## Current V2 Model

- Model: `TemporalRiskPredictor`
- Architecture: GRU sequence predictor
- Input window: 24 frames
- Feature dimension: 23
- Output: `[forklift_target, dropzone_target]`
- Scores: `SAFE < 0.4`, `0.4 <= WARNING < 0.8`, `DANGER >= 0.8`
- Checkpoint:
  `model/fusion_v2/checkpoints_geometry_future/best.pt`

### Geometry Future Labels

The `geometry_future` label mode creates labels from absolute coordinates.

| Target | WARNING | DANGER | Future Horizon |
| --- | ---: | ---: | ---: |
| Forklift | worker to forklift/front-hazard <= 2.4 m | <= 1.25 m | next 12 frames |
| DropZone | worker to dropzone <= 2.8 m | <= 2.0 m | next 12 frames |

The future horizon means a frame is promoted to `WARNING` or `DANGER` when the
risk appears within the next 12 frames.  This makes V2 learn early prediction,
not only current-frame contact.

## Latest Independent Run

- Existing recorded scenario sources: 7
- Generated geometry scenario sources: 450
- Total scenario sources: 457
- Dataset windows: 89,176
- Window size: 24 frames
- Stride: 2
- Augmentation: 1 noisy copy per window
- Noise std: 0.02
- Dataset:
  `model/fusion_v2/data/fusion_v2_geometry_future_dataset.npz`

### Label Balance

| Target | SAFE | WARNING | DANGER |
| --- | ---: | ---: | ---: |
| Forklift | 76,090 (85.3%) | 3,560 (4.0%) | 9,526 (10.7%) |
| DropZone | 59,834 (67.1%) | 4,524 (5.1%) | 24,818 (27.8%) |

### Full Dataset Evaluation

| Target | Accuracy | Danger Precision | Danger Recall | Danger F1 |
| --- | ---: | ---: | ---: | ---: |
| Overall class | 99.41% | - | - | - |
| Forklift | 99.68% | 98.13% | 98.92% | 98.52% |
| DropZone | 99.71% | 99.50% | 99.45% | 99.48% |

### Scenario Review

| Scenario | Main Risk | Reference First Danger | V2 First Danger | Note |
| --- | --- | ---: | ---: | --- |
| `scenario_01_user_current` | W02 vs Forklift | frame 105 | frame 102 | V2 predicts slightly earlier. |
| `scenario_02_swapped_positions` | W01 vs Forklift | frame 23 | frame 23 | V2 matches the geometry reference. |
| `scenario_03_opposite_worker` | W02 vs Forklift | frame 99 | frame 96 | V2 predicts slightly earlier. |
| `scenario_04_box_dropzone` | W01 vs DropZone | frame 103 | frame 102 | Box/DropZone risk is detected. |
| `scenario_04_box_dropzone` | W02 vs DropZone | frame 33 | frame 31 | V2 predicts slightly earlier. |

Some V1 and V2 decisions intentionally disagree.  V1 can trigger earlier from
TTC/front-hazard heuristics, while this V2 run is trained from future geometry
thresholds.  That disagreement is useful for comparing rule-based and learned
risk engines.

## Commands

```bash
conda activate venv

python -m model.fusion_v2.generate_scenarios \
  --output-root model/fusion_v2/generated_scenarios_geometry \
  --count-per-kind 45 \
  --seed 20260528 \
  --clean

python -m model.fusion_v2.dataset \
  --input-root simulation/Recordings/collision_scenarios model/fusion_v2/generated_scenarios_geometry \
  --output model/fusion_v2/data/fusion_v2_geometry_future_dataset.npz \
  --window-size 24 \
  --stride 2 \
  --augment 1 \
  --noise-std 0.02 \
  --label-mode geometry_future \
  --future-horizon-frames 12 \
  --forklift-danger-m 1.25 \
  --forklift-warning-m 2.4 \
  --dropzone-danger-m 2.0 \
  --dropzone-warning-m 2.8

python -m model.fusion_v2.train \
  --dataset model/fusion_v2/data/fusion_v2_geometry_future_dataset.npz \
  --output-dir model/fusion_v2/checkpoints_geometry_future \
  --epochs 20 \
  --batch-size 256 \
  --seed 42

python -m model.fusion_v2.evaluate \
  --dataset model/fusion_v2/data/fusion_v2_geometry_future_dataset.npz \
  --checkpoint model/fusion_v2/checkpoints_geometry_future/best.pt \
  --output-dir model/fusion_v2/reports/geometry_future

python -m model.fusion_v2.test_scenarios \
  --checkpoint model/fusion_v2/checkpoints_geometry_future/best.pt \
  --output-dir model/fusion_v2/reports/scenario_tests_geometry_future \
  --label-mode geometry_future \
  --future-horizon-frames 12
```

## Next Work

1. Export Unity ground-truth positions and event labels to remove dependency
   on perception-derived coordinates during training.
2. Add more hard-negative and near-miss scenarios so `WARNING` cases are less
   underrepresented.
3. Compare V1 and V2 side-by-side on fixed hold-out videos before adding a
   runtime switch such as `--risk-engine v1|v2`.
