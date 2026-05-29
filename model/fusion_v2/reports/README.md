# Fusion V2 Evaluation Report

## What Was Tested

V2 was trained as a separate deep-learning risk prediction track while keeping
V1 unchanged.

```text
V1: YOLO/Pose -> BEV coordinates -> rule-based Warning/Danger
V2: YOLO/Pose -> BEV coordinates -> recent coordinate window -> GRU risk model
```

The current V2 model learns from V1 diagnostic CSVs.  In other words, V1 is the
teacher and V2 is trained to reproduce the V1 risk decision pattern.

## Dataset

- Existing V1/Unity diagnostic scenarios: 7
- Generated V2 coordinate scenarios: 210
- Total scenario sources: 217
- Total windows: 42,136
- Sequence length: 24 frames
- Feature dimension: 23

## Label Distribution

| Target | SAFE | WARNING | DANGER |
| --- | ---: | ---: | ---: |
| Forklift | 32,336 (76.7%) | 5,858 (13.9%) | 3,942 (9.4%) |
| DropZone | 31,440 (74.6%) | 1,940 (4.6%) | 8,756 (20.8%) |

## V1 vs V2

| Target | Accuracy | Danger Precision | Danger Recall | Danger F1 |
| --- | ---: | ---: | ---: | ---: |
| Overall class | 98.48% | - | - | - |
| Forklift | 99.22% | 95.98% | 95.64% | 95.81% |
| DropZone | 99.53% | 98.86% | 98.88% | 98.87% |

## Interpretation

This result is good for a first V2 baseline because the GRU model learned the
V1 coordinate-based decision pattern with high agreement.

It is not yet proof that V2 is better than V1.  The current labels are generated
from V1, not from human-reviewed incident labels or Unity ground-truth collision
events.  The next step is to add independent validation scenarios and compare
the first Warning/Danger frame between V1 and V2.
