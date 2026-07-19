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

def moving_variance(data, window_size=100):
    if len(data) < window_size:
        return np.zeros_like(data)
    d = np.array(data)
    d2 = d**2
    mean = np.convolve(d, np.ones(window_size)/window_size, mode='valid')
    mean_of_sq = np.convolve(d2, np.ones(window_size)/window_size, mode='valid')
    var = mean_of_sq - mean**2
    return np.maximum(var, 0)

def plot_multicomparison(runs, labels, out_dir, tag, ylabel, title, filename, smooth=1, plot_variance=False):
    plt.figure(figsize=(8, 5))
    colors = ['gray', 'blue', 'red', 'green', 'orange']
    has_data = False
    
    for i, run_dir in enumerate(runs):
        steps, vals = get_scalars(run_dir, tag)
        if not steps:
            print(f"Warning: No data found for {tag} in {run_dir}")
            continue
        has_data = True
        
        if "Success_Rate" in tag or "_Pct" in tag:
            filt = [(s, v) for s, v in zip(steps, vals) if v > 0.0 or s == 0]
            if filt:
                steps, vals = zip(*filt)
            vals = np.array(vals)
            if len(vals) > 0 and np.max(vals) <= 1.05:
                vals = vals * 100.0

        if plot_variance:
            if len(vals) >= smooth:
                var_vals = moving_variance(vals, smooth)
                plt.plot(steps[smooth-1:], var_vals, label=labels[i], color=colors[i % len(colors)], linewidth=2, alpha=0.8 if i==0 else 1.0)
            else:
                plt.plot(steps, np.zeros_like(vals), label=labels[i], color=colors[i % len(colors)], linewidth=2)
        else:
            if smooth > 1 and len(vals) >= smooth:
                plt.plot(steps[smooth-1:], moving_average(vals, smooth), label=labels[i], color=colors[i % len(colors)], linewidth=2, alpha=0.8 if i==0 else 1.0)
            else:
                plt.plot(steps, vals, label=labels[i], color=colors[i % len(colors)], linewidth=2, alpha=0.8 if i==0 else 1.0)
            
    if not has_data:
        plt.close()
        return

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
    parser = argparse.ArgumentParser(description="Plot comparison between runs")
    parser.add_argument("--runs", type=str, nargs='+', required=True, help="Paths to run directories")
    parser.add_argument("--labels", type=str, nargs='+', required=True, help="Labels for the runs")
    parser.add_argument("--outdir", type=str, default="plots/comparison", help="Output directory")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for saved filenames")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving comparison plots to {args.outdir}")

    if len(args.runs) != len(args.labels):
        print("Error: Number of runs and labels must match.")
        exit(1)

    plot_multicomparison(args.runs, args.labels, args.outdir, 
                         'Rewards/Avg_Env_Reward', "Average Extrinsic Reward", "Average Extrinsic Reward", 
                         f"{args.prefix}rewards.png", smooth=50)

    plot_multicomparison(args.runs, args.labels, args.outdir, 
                         'TD_Error/Abs_Mean', "TD Error Variance", "Temporal Difference Error Variance", 
                         f"{args.prefix}td_error.png", smooth=100, plot_variance=True)

    # Subtasks
    subtasks = ["Wood", "Stone", "Pickaxe", "Iron", "Sword", "Armor", "Bridge", "Enemy", "Gold"]
    for t in subtasks:
        plot_multicomparison(args.runs, args.labels, args.outdir, 
                        f'Subtasks/{t}_Pct', "Success Rate (%)", f"{t} Success Rate", 
                        f"{args.prefix}subtask_{t}.png", smooth=50)
