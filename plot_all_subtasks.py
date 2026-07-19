import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_max_subtasks(run_dir):
    event_files = [f for f in os.listdir(run_dir) if "tfevents" in f]
    if not event_files: return {}
    ea = EventAccumulator(os.path.join(run_dir, event_files[0]))
    ea.Reload()
    
    subtasks = ["Wood", "Stone", "Pickaxe", "Iron", "Sword", "Armor", "Bridge", "Enemy", "Gold"]
    results = {}
    for t in subtasks:
        tag = f"Subtasks/{t}_Pct"
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            results[t] = max([e.value for e in events]) * 100
        else:
            results[t] = 0.0
    return results

def plot_grouped_bars(run_dirs, labels, out_file, title):
    subtasks = ["Wood", "Stone", "Pickaxe", "Iron", "Sword", "Armor", "Bridge", "Enemy", "Gold"]
    data = []
    for d in run_dirs:
        data.append(get_max_subtasks(d))
    
    x = np.arange(len(subtasks))
    width = 0.8 / len(labels)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['#7f7f7f', '#1f77b4', '#d62728']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        y = [data[i][t] for t in subtasks]
        offset = (i - len(labels)/2 + 0.5) * width
        ax.bar(x + offset, y, width, label=label, color=color)
        
    ax.set_ylabel('Max Success Rate (%)')
    ax.set_title(title, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(subtasks)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    baseline = "runs/v3_LLMDynamic_Fair_NoDAG_qwen2.5-7b_E128_s42_20260630-075824"
    no_ema = "runs/v3_Ablation_NoEMA_qwen2.5-7b_E128_s42_20260717-165225"
    furnace = "runs/v3_EnvShift_Furnace_qwen2.5-7b_E128_s42_20260718-061057"
    zylorg = "runs/v3_EnvShift_Zylorg_qwen2.5-7b_E128_s42_20260718-153853"
    
    os.makedirs("thesis/plots/ablations_and_shifts", exist_ok=True)
    plot_grouped_bars([baseline, no_ema], ["Qwen (Baseline)", "Qwen (No-EMA)"], "thesis/plots/ablations_and_shifts/ablation_all_subtasks_bar.png", "Ablation: Max Subtask Completion Rates")
    
    plot_grouped_bars([baseline, furnace, zylorg], ["Qwen (Baseline)", "Qwen (Furnace)", "Qwen (Zylorg)"], "thesis/plots/ablations_and_shifts/shift_all_subtasks_bar.png", "Semantic Shift: Max Subtask Completion Rates")
