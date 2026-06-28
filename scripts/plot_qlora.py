import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_qlora_scalars(log_dir, tag):
    log_files = sorted(glob.glob(os.path.join(log_dir, "*")))
    all_steps, all_vals = [], []
    for lf in log_files:
        if os.path.isdir(lf):
            continue
        ea = event_accumulator.EventAccumulator(lf)
        ea.Reload()
        if tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            all_steps.extend([e.step for e in events])
            all_vals.extend([e.value for e in events])
    
    if not all_steps:
        return [], []
    
    sorted_pairs = sorted(zip(all_steps, all_vals))
    steps, vals = zip(*sorted_pairs)
    return steps, vals

def plot_qlora_training(log_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    steps_loss, vals_loss = get_qlora_scalars(log_dir, 'train/loss')
    steps_acc, vals_acc = get_qlora_scalars(log_dir, 'train/mean_token_accuracy')
    
    if not steps_loss:
        print("Warning: No QLoRA training data found!")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(steps_loss, vals_loss, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    if steps_acc:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Mean Token Accuracy', color=color)
        ax2.plot(steps_acc, vals_acc, color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 1.05])

    plt.title('QLoRA Adapter Fine-Tuning Performance')
    fig.tight_layout()
    out_file = os.path.join(out_dir, "QLoRA_Training_Performance.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    log_dir = "data/models/qlora_adapter/logs"
    out_dir = "plots/qlora"
    print(f"Plotting QLoRA metrics from {log_dir}")
    plot_qlora_training(log_dir, out_dir)
