"""Regenerate all HRL ablation plots for the 3 clean runs FASTER."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Run paths ──
RUNS = {
    'No-LoRA Base':      'runs/v3_HRL_Std_Ollama_E128_s42_20260721-220545',
    'QLoRA + Reflection':'runs/v3_HRL_Std_LoRA_E128_s42_20260722-120127',
    'QLoRA – No Refl.':  'runs/v3_HRL_Std_LoRA_E128_s42_20260721-094612',
}
JSONL = {
    'No-LoRA Base':       'data/trajectories/update_metrics_20260721-220545.jsonl',
    'QLoRA + Reflection': 'data/trajectories/update_metrics_20260722-120127.jsonl',
    'QLoRA – No Refl.':   'data/trajectories/update_metrics_20260721-094612.jsonl',
}
OUTDIR = 'plots/hrl_reflection_ablation'
COLORS = {'No-LoRA Base': '#888888', 'QLoRA + Reflection': '#2196F3', 'QLoRA – No Refl.': '#E53935'}
os.makedirs(OUTDIR, exist_ok=True)

# ── Helpers ──
def smooth(data, w=50):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='valid')

def moving_var(data, w=100):
    if len(data) < w:
        return np.zeros_like(data)
    d = np.array(data)
    m = np.convolve(d, np.ones(w)/w, mode='valid')
    m2 = np.convolve(d**2, np.ones(w)/w, mode='valid')
    return np.maximum(m2 - m**2, 0)

# ── Load all TensorBoard data ONCE ──
print("Loading TensorBoard data...")
tb_data = {}
for name, path in RUNS.items():
    ea = EventAccumulator(path)
    ea.Reload()
    tb_data[name] = ea
    print(f"  Loaded {name}")

def get_tb(name, tag):
    ea = tb_data[name]
    if tag not in ea.Tags().get('scalars', []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]

# ═══════════════════════════════════════════
# 1. Subtask Completion Rate Plots (9 plots)
# ═══════════════════════════════════════════
subtasks = ['Wood', 'Stone', 'Pickaxe', 'Iron', 'Sword', 'Armor', 'Bridge', 'Enemy', 'Gold']
for task in subtasks:
    tag = f'Subtasks/{task}_Pct'
    plt.figure(figsize=(8, 5))
    for name in RUNS:
        steps, vals = get_tb(name, tag)
        if not steps:
            continue
        vals = np.array(vals)
        # Cap at 100% (Gold has a logging bug at 10000%)
        vals = np.clip(vals, 0, 1.0)
        vals = vals * 100.0
        if len(vals) >= 50:
            s = smooth(vals, 50)
            plt.plot(steps[49:], s, label=name, color=COLORS[name], linewidth=2)
        else:
            plt.plot(steps, vals, label=name, color=COLORS[name], linewidth=2)
    plt.xlabel("Environment Steps")
    plt.ylabel("Success Rate (%)")
    plt.title(f"{task} Completion Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"hrl_ablation_subtask_{task}.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=150)
    plt.close()
    print(f"Saved {fname}")

# ═══════════════════════════════════════════
# 2. Reward Plot
# ═══════════════════════════════════════════
plt.figure(figsize=(8, 5))
for name in RUNS:
    steps, vals = get_tb(name, 'Rewards/Avg_Env_Reward')
    if not steps:
        continue
    if len(vals) >= 50:
        s = smooth(vals, 50)
        plt.plot(steps[49:], s, label=name, color=COLORS[name], linewidth=2)
    else:
        plt.plot(steps, vals, label=name, color=COLORS[name], linewidth=2)
plt.xlabel("Environment Steps")
plt.ylabel("Average Extrinsic Reward")
plt.title("Average Extrinsic Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'hrl_ablation_rewards.png'), dpi=150)
plt.close()
print("Saved hrl_ablation_rewards.png")

# ═══════════════════════════════════════════
# 3. TD Error Variance Plot
# ═══════════════════════════════════════════
plt.figure(figsize=(8, 5))
for name in RUNS:
    steps, vals = get_tb(name, 'TD_Error/Abs_Mean')
    if not steps:
        continue
    if len(vals) >= 100:
        var = moving_var(vals, 100)
        plt.plot(steps[99:], var, label=name, color=COLORS[name], linewidth=2)
    else:
        plt.plot(steps, np.zeros_like(vals), label=name, color=COLORS[name], linewidth=2)
plt.xlabel("Environment Steps")
plt.ylabel("TD Error Variance")
plt.title("Temporal Difference Error Variance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'hrl_ablation_td_error.png'), dpi=150)
plt.close()
print("Saved hrl_ablation_td_error.png")

# ═══════════════════════════════════════════
# 4. KL Divergence Spikes Plot (from JSONL)
# ═══════════════════════════════════════════
plt.figure(figsize=(8, 5))
for name, jpath in JSONL.items():
    updates = []
    kl_vals = []
    with open(jpath) as f:
        for line in f:
            d = json.loads(line)
            updates.append(d['update'] * 32768)  # 128 envs * 256 rollout
            kl_vals.append(d['max_kl'])
    if len(kl_vals) >= 5:
        s = smooth(kl_vals, 5)
        plt.plot(updates[4:], s, label=name, color=COLORS[name], linewidth=1.5)
    else:
        plt.plot(updates, kl_vals, label=name, color=COLORS[name], linewidth=1.5)
plt.axhline(y=0.015, color='r', linestyle='--', linewidth=1, label='KL Threshold ($\\delta=0.015$)')
plt.xlabel("Environment Steps")
plt.ylabel("Max Approx KL Divergence")
plt.title("Maximum Approximate KL Divergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'hrl_ablation_kl_spikes.png'), dpi=150)
plt.close()
print("Saved hrl_ablation_kl_spikes.png")

print("\nAll plots regenerated successfully!")
