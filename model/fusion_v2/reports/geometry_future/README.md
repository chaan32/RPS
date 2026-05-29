# Fusion V2 Geometry Future Report

This report records the first V2 run where labels are generated independently
from V1 risk outputs.

## What Changed

The previous V2 baseline used V1 outputs as teacher labels.  This run uses
`label_mode=geometry_future`, which creates labels from absolute coordinate
geometry:

- worker position
- forklift center position
- forklift front-hazard position
- dropzone position
- object existence flags
- future risk within the next 12 frames

The label builder does not use V1 columns such as `forklift_risk`,
`dropzone_risk`, `early_level`, or `dropzone_forced`.

## Dataset

| Item | Value |
| --- | ---: |
| Scenario sources | 457 |
| Windows | 89,176 |
| Window size | 24 frames |
| Feature dimension | 23 |
| Future horizon | 12 frames |
| Forklift warning / danger | 2.4 m / 1.25 m |
| DropZone warning / danger | 2.8 m / 2.0 m |

## Metrics

| Target | Accuracy | Danger Precision | Danger Recall | Danger F1 |
| --- | ---: | ---: | ---: | ---: |
| Overall class | 99.41% | - | - | - |
| Forklift | 99.68% | 98.13% | 98.92% | 98.52% |
| DropZone | 99.71% | 99.50% | 99.45% | 99.48% |

Source files:

- `model/fusion_v2/data/fusion_v2_geometry_future_dataset.npz`
- `model/fusion_v2/checkpoints_geometry_future/best.pt`
- `model/fusion_v2/reports/geometry_future/v1_v2_comparison_summary.json`
- `model/fusion_v2/reports/geometry_future/v1_v2_predictions.csv`

## Scenario Videos

| Scenario | Output Video |
| --- | --- |
| `scenario_01_user_current` | `model/fusion_v2/reports/scenario_tests_geometry_future/scenario_01_user_current/scenario_01_user_current_v2_test.mp4` |
| `scenario_02_swapped_positions` | `model/fusion_v2/reports/scenario_tests_geometry_future/scenario_02_swapped_positions/scenario_02_swapped_positions_v2_test.mp4` |
| `scenario_03_opposite_worker` | `model/fusion_v2/reports/scenario_tests_geometry_future/scenario_03_opposite_worker/scenario_03_opposite_worker_v2_test.mp4` |
| `scenario_04_box_dropzone` | `model/fusion_v2/reports/scenario_tests_geometry_future/scenario_04_box_dropzone/scenario_04_box_dropzone_v2_test.mp4` |

## Caveat

This is independent from V1's risk decision logic, but not fully independent
from the V1 perception pipeline.  The input coordinates are still produced by
YOLO/Pose and Homography.  To make V2 fully simulator-ground-truth based, Unity
should export object positions and event labels directly.
