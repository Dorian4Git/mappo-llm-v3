import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_scalars(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_metric(run_dir, out_dir, tag, ylabel, title, filename, smooth=1):
    steps, vals = get_scalars(run_dir, tag)
    if not steps:
        print(f"Warning: No data found for {tag}")
        return

    # Filter out exactly 0.0 values for success rates which are artifacts of missing episodes in the batch
    if "Success_Rate" in tag:
        filt = [(s, v) for s, v in zip(steps, vals) if v > 0.0 or s == 0]
        if filt:
            steps, vals = zip(*filt)

    plt.figure(figsize=(8, 5))
    
    if smooth > 1:
        plt.plot(steps[smooth-1:], moving_average(vals, smooth), label=tag, color='blue', linewidth=2)
    else:
        plt.plot(steps, vals, label=tag, color='blue', linewidth=2)
        
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()
    print(f"Saved {filename}")

def plot_subtasks(run_dir, out_dir, smooth=10):
    tags = ["Wood", "Stone", "Pickaxe", "Iron", "Sword", "Armor", "Bridge", "Enemy", "Gold"]
    
    plt.figure(figsize=(10, 6))
    
    for tag in tags:
        full_tag = f"Subtasks/{tag}_Pct"
        steps, vals = get_scalars(run_dir, full_tag)
        if steps:
            filt = [(s, v) for s, v in zip(steps, vals) if v > 0.0 or s == 0]
            if filt:
                steps, vals = zip(*filt)
                
            # Fix scaling bug: Gold is logged as 0-100, others are 0-1.
            vals = np.array(vals)
            if np.max(vals) <= 1.05:
                vals = vals * 100.0
                
            if smooth > 1:
                plt.plot(steps[smooth-1:], moving_average(vals, smooth), label=tag, linewidth=2)
            else:
                plt.plot(steps, vals, label=tag, linewidth=2)
                
    plt.xlabel("Environment Steps")
    plt.ylabel("Completion Percentage")
    plt.title("Subtask Completion Rates Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Subtasks_Completion.png"))
    plt.close()
    print("Saved Subtasks_Completion.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from a TensorBoard run directory")
    parser.add_argument("run_dir", type=str, help="Path to the run directory (e.g., runs/v3_HRL_...)")
    args = parser.parse_args()

    run_dir = args.run_dir
    run_name = os.path.basename(os.path.normpath(run_dir))
    
    out_dir = os.path.join("plots", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving plots to {out_dir}")

    plot_metric(run_dir, out_dir, 'Rewards/Avg_Env_Reward', "Average Extrinsic Reward", "Average Extrinsic Reward over Training", "Rewards_Avg_Env_Reward.png", smooth=20)
    plot_subtasks(run_dir, out_dir, smooth=50)
    
    # If using success rate logging
    plot_metric(run_dir, out_dir, 'Episodes/Success_Rate', "Success Rate", "Gold Mining Success Rate", "Episodes_Success_Rate.png", smooth=10)
