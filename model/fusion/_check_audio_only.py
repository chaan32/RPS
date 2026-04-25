"""오디오 단독 효과 검증: 위치 안전 + 오디오만 이상."""
import os
import numpy as np
import torch

from scenario_generator import Scenario, RATE, DURATION, N_STEPS, still, linear, audio_trace, crane_seq
from inference import load_model, predict_scenario

ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")
model = load_model(ckpt, device="cpu")

# 안전 위치 + 오디오 spike만 있는 가상 시나리오
def make_test(audio_spike=None, name="test"):
    return Scenario(
        name=name,
        forklift=linear([-2, 0], [-1.8, 0]),  # 거의 정지, 멀리
        worker1=still([-0.3, 2.7]),            # 멀리 위쪽
        crane_state=crane_seq(0),
        audio=audio_trace(base=0.05, spike=audio_spike),
        labels=None,
    )

print("[Test 1: 안전 위치 + 오디오 정상 (0.05)]")
s_normal = make_test(audio_spike=None, name="audio_normal")
risk = predict_scenario(model, s_normal)
print(f"  forklift risk @ t=10,15,20s: "
      f"{risk[50,0,0]:.3f}, {risk[75,0,0]:.3f}, {risk[99,0,0]:.3f}")
print(f"  dropzone risk @ t=10,15,20s: "
      f"{risk[50,0,1]:.3f}, {risk[75,0,1]:.3f}, {risk[99,0,1]:.3f}")
print()

print("[Test 2: 안전 위치 + 오디오 mid spike (0.5)]")
s_mid = make_test(audio_spike=(8, 20, 0.5), name="audio_mid")
risk = predict_scenario(model, s_mid)
print(f"  forklift risk @ t=10,15,20s: "
      f"{risk[50,0,0]:.3f}, {risk[75,0,0]:.3f}, {risk[99,0,0]:.3f}")
print(f"  dropzone risk @ t=10,15,20s: "
      f"{risk[50,0,1]:.3f}, {risk[75,0,1]:.3f}, {risk[99,0,1]:.3f}")
print()

print("[Test 3: 안전 위치 + 오디오 strong spike (0.9)]")
s_strong = make_test(audio_spike=(8, 20, 0.9), name="audio_strong")
risk = predict_scenario(model, s_strong)
print(f"  forklift risk @ t=10,15,20s: "
      f"{risk[50,0,0]:.3f}, {risk[75,0,0]:.3f}, {risk[99,0,0]:.3f}")
print(f"  dropzone risk @ t=10,15,20s: "
      f"{risk[50,0,1]:.3f}, {risk[75,0,1]:.3f}, {risk[99,0,1]:.3f}")
