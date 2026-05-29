# Fusion V2 Scenario Video Test

## Test Scope

The Fusion V2 model was tested on the four current recorded validation
scenarios:

1. `scenario_01_user_current`
2. `scenario_02_swapped_positions`
3. `scenario_03_opposite_worker`
4. `scenario_04_box_dropzone`

V2 consumes the V1-produced `fusion_risk.csv` coordinate sequence, not raw
camera images.  The review videos therefore show the recorded cam1/cam2 frames
with a side panel comparing V1 teacher risk and V2 predicted risk.

## Summary

| Scenario | Workers | Forklift Class Acc | Forklift Danger F1 | DropZone Class Acc | DropZone Danger F1 | Main Issue |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| scenario_01_user_current | W01, W02 | 0.798 | 0.933 | 1.000 | 0.000 | Forklift danger is detected, but warning-level class matching is still weak. |
| scenario_02_swapped_positions | W01, W02 | 0.691 | 0.780 | 1.000 | 0.000 | W02 forklift danger is detected 6 frames later than V1. |
| scenario_03_opposite_worker | W01, W02 | 0.897 | 0.444 | 1.000 | 0.000 | W02 forklift danger is 20 frames late, and W01 warning is missed. |
| scenario_04_box_dropzone | W01, W02 | 1.000 | 0.000 | 0.722 | 0.261 | W02 DropZone danger is detected, but W01 DropZone danger remains below threshold. |

## Interpretation

V2 learned the broad V1 decision pattern, but it is not ready to replace V1.
The most important failures are late/missed danger timing in scenario 3 and
weak DropZone danger recall for W01 in scenario 4.

The next training set should add more hard examples around:

- Opposite-side worker approach near the blind corner.
- Borderline Warning-to-Danger forklift transitions.
- DropZone danger cases where the box only partially overlaps the worker path.
- Per-worker DropZone examples for both W01 and W02.

## Outputs

- `scenario_v2_summary.json`: aggregated scenario summary.
- `*/v2_summary.json`: per-scenario metrics and first-event frames.
- `*/v2_predictions.csv`: per-window V1 target and V2 prediction.
- `*/*_v2_test.mp4`: cam1/cam2 review video with V1/V2 risk side panel.
