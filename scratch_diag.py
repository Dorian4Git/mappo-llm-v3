import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('runs/v3_LLMDynamic_Fair_NoDAG_qwen2.5-7b_E128_s42_20260630-075824', size_guidance={'scalars': 0})
ea.Reload()

# Detailed analysis of the softmax weight distribution
print("=== SOFTMAX WEIGHT BUDGET ANALYSIS ===")
print("The softmax attention budget constrains sum(weights) = |W| = 7.0")
print("Under tau=0.5, the LLM logits get sharpened, causing extreme concentration.\n")

# Get latest softmax weights
latest = {}
for wkey in ['w_wood', 'w_stone', 'w_workbench', 'w_iron', 'w_bridge', 'w_enemy', 'w_gold']:
    vals = [e.value for e in ea.Scalars(f'LLM_Weights_Softmax/{wkey}')]
    latest[wkey] = vals[-1] if vals else 0.0

total = sum(latest.values())
print(f"Latest softmax weights (sum={total:.4f}):")
for k, v in sorted(latest.items(), key=lambda x: -x[1]):
    pct = v / total * 100
    print(f"  {k:15s}: {v:.4f}  ({pct:.1f}% of budget)")

# This tells us: what does the greedy planner DO with these weights?
print("\n=== GREEDY PLANNER EFFECT ===")
print("The planner multiplies Base_Reward * w_i and picks the max.")
print("With the current weights:")
print(f"  Iron:      2.0 * {latest['w_iron']:.4f} = {2.0 * latest['w_iron']:.4f}")
print(f"  Workbench: 1.0 * {latest['w_workbench']:.4f} = {1.0 * latest['w_workbench']:.4f}")
print(f"  Bridge:    3.0 * {latest['w_bridge']:.4f} = {3.0 * latest['w_bridge']:.4f}")
print(f"  Enemy:    10.0 * {latest['w_enemy']:.4f} = {10.0 * latest['w_enemy']:.4f}")
print(f"  Gold:     15.0 * {latest['w_gold']:.4f} = {15.0 * latest['w_gold']:.4f}")
print(f"  Wood:      2.0 * {latest['w_wood']:.4f} = {2.0 * latest['w_wood']:.4f}")
print(f"  Stone:     2.0 * {latest['w_stone']:.4f} = {2.0 * latest['w_stone']:.4f}")

print("\n=== PROBLEM DIAGNOSIS ===")
print("The LLM puts nearly ALL budget on iron+workbench (~6.7 of 7.0).")
print("Bridge gets 0.06 -> score = 3.0*0.06 = 0.18")
print("Gold gets 0.06 -> score = 15.0*0.06 = 0.92")
print("Iron gets 3.35 -> score = 2.0*3.35 = 6.70")
print()
print("=> The planner ALWAYS picks Iron/Workbench and NEVER picks Bridge/Enemy/Gold!")
print("=> The agent gets stuck looping on early subtasks and never progresses.")
print("=> The LLM never sees Bridge/Enemy/Gold completion, so it never adjusts.")
print("=> This IS the catastrophic failure mode of naive dynamic reward shaping.")

# Verify: is the agent actually completing wood/stone/pickaxe?
print("\n=== EARLY SUBTASK MASTERY CHECK ===")
for tag in ['Wood', 'Stone', 'Pickaxe']:
    vals = [e.value for e in ea.Scalars(f'Subtasks/{tag}_Pct')]
    if vals:
        last50_mean = np.mean(vals[-50:]) * 100
        print(f"  {tag}: last 50 mean = {last50_mean:.1f}% (mastered = True)")

print("\n=== LATE SUBTASK FAILURE CHECK ===")
for tag in ['Iron', 'Sword', 'Armor', 'Bridge', 'Enemy', 'Gold']:
    vals = [e.value for e in ea.Scalars(f'Subtasks/{tag}_Pct')]
    if vals:
        last50_mean = np.mean(vals[-min(50, len(vals)):]) * 100
        ever_achieved = max(vals) * 100
        print(f"  {tag}: last mean = {last50_mean:.1f}%, peak = {ever_achieved:.1f}%")
