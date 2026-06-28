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

def plot_comparison(run1_dir, run1_label, run2_dir, run2_label, out_dir, tag, ylabel, title, filename, smooth=1):
    steps1, vals1 = get_scalars(run1_dir, tag)
    steps2, vals2 = get_scalars(run2_dir, tag)
    
    if not steps1 and not steps2:
        print(f"Warning: No data found for {tag}")
        return

    # Filter out exactly 0.0 values for success rates which are artifacts of missing episodes in the batch
    if "Success_Rate" in tag or "_Pct" in tag:
        filt1 = [(s, v) for s, v in zip(steps1, vals1) if v > 0.0 or s == 0]
        if filt1:
            steps1, vals1 = zip(*filt1)
        filt2 = [(s, v) for s, v in zip(steps2, vals2) if v > 0.0 or s == 0]
        if filt2:
            steps2, vals2 = zip(*filt2)

    plt.figure(figsize=(8, 5))
    
    if steps1:
        if smooth > 1:
            plt.plot(steps1[smooth-1:], moving_average(vals1, smooth), label=run1_label, color='gray', alpha=0.8, linewidth=2)
        else:
            plt.plot(steps1, vals1, label=run1_label, color='gray', alpha=0.8, linewidth=2)
            
    if steps2:
        if smooth > 1:
            plt.plot(steps2[smooth-1:], moving_average(vals2, smooth), label=run2_label, color='blue', linewidth=2)
        else:
            plt.plot(steps2, vals2, label=run2_label, color='blue', linewidth=2)
        
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison between two runs")
    parser.add_argument("run1", type=str, help="Path to Baseline Run")
    parser.add_argument("run2", type=str, help="Path to New Run")
    parser.add_argument("--label1", type=str, default="Baseline", help="Label for run 1")
    parser.add_argument("--label2", type=str, default="New Run", help="Label for run 2")
    parser.add_argument("--outdir", type=str, default="plots/comparison", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving comparison plots to {args.outdir}")

    # Core Performance Metrics
    plot_comparison(args.run1, args.label1, args.run2, args.label2, args.outdir, 
                    'Rewards/Avg_Env_Reward', "Average Extrinsic Reward", "Average Extrinsic Reward Comparison", "Rewards_Avg_Env_Reward.png", smooth=50)
                    
    # Subtasks
    plot_comparison(args.run1, args.label1, args.run2, args.label2, args.outdir, 
                    'Subtasks/Gold_Pct', "Success Rate", "Gold Mining Success Rate", "Subtasks_Gold_Pct.png", smooth=50)
    plot_comparison(args.run1, args.label1, args.run2, args.label2, args.outdir, 
                    'Subtasks/Enemy_Pct', "Success Rate", "Enemy Defeat Success Rate", "Subtasks_Enemy_Pct.png", smooth=50)
    plot_comparison(args.run1, args.label1, args.run2, args.label2, args.outdir, 
                    'Subtasks/Bridge_Pct', "Success Rate", "Bridge Building Success Rate", "Subtasks_Bridge_Pct.png", smooth=50)
