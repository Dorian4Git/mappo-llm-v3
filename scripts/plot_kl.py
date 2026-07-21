import json
import os
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(8, 5))
colors = ['gray', 'blue', 'red', 'green', 'orange']
labels = ['No-LoRA Base', 'QLoRA (With Reflection)', 'QLoRA (No Reflection)']
runs = ['20260720-085701', '20260720-190702', '20260721-094612']
has_kl_data = False

for i, timestamp in enumerate(runs):
    jsonl_path = os.path.join('data', 'trajectories', f'update_metrics_{timestamp}.jsonl')
    steps = []
    vals = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'max_kl' in data:
                        steps.append(data.get('update', 0) * 32768)
                        vals.append(data['max_kl'])
                except:
                    pass
                    
    if not steps:
        continue
    has_kl_data = True
    
    smooth = 5
    if len(vals) >= smooth:
        smoothed_vals = moving_average(vals, smooth)
        plt.plot(steps[smooth-1:], smoothed_vals, label=labels[i], color=colors[i % len(colors)], linewidth=1.5, alpha=0.8 if i==0 else 1.0)
    else:
        plt.plot(steps, vals, label=labels[i], color=colors[i % len(colors)], linewidth=1.5)

if has_kl_data:
    plt.axhline(y=0.015, color='r', linestyle='--', label='KL Guardrail Threshold')
    plt.xlabel('Environment Steps')
    plt.ylabel('Max Approx KL')
    plt.title('Maximum Approximate KL Divergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('plots/hrl_reflection_ablation', exist_ok=True)
    plt.savefig('plots/hrl_reflection_ablation/hrl_ablation_kl_spikes.png')
    print('Saved hrl_ablation_kl_spikes.png')
