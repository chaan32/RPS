"""오디오가 실제로 모델 출력에 영향 주는지 검증."""
from scenarios_synthetic import build_synthetic_24
from inference import load_model, predict_scenario
import os, numpy as np

ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")
model = load_model(ckpt, device="cpu")
sc = build_synthetic_24()

s05 = next(s for s in sc if s.name == "s05_fork_slow_approach_static_worker")
s14 = next(s for s in sc if s.name == "s14_fork_approach_with_audio_spike")

print("[s05 - no audio spike]")
print(f"  audio @ t=10,15,20s: {[f'{s05.audio[i]:.2f}' for i in [50, 75, 99]]}")
risk5 = predict_scenario(model, s05)
print(f"  forklift risk @ t=15,20s: {risk5[75,0,0]:.3f}, {risk5[99,0,0]:.3f}")

print("\n[s14 - SAME positions + audio spike at t=10-20s]")
print(f"  audio @ t=10,15,20s: {[f'{s14.audio[i]:.2f}' for i in [50, 75, 99]]}")
risk14 = predict_scenario(model, s14)
print(f"  forklift risk @ t=15,20s: {risk14[75,0,0]:.3f}, {risk14[99,0,0]:.3f}")

print("\n[diff: s14 - s05]")
print(f"  forklift: {risk14[75,0,0]-risk5[75,0,0]:+.3f} @ t=15s,  {risk14[99,0,0]-risk5[99,0,0]:+.3f} @ t=20s")
