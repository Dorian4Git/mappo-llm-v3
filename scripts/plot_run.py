import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_scalars(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={'scalars': 0})
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

    # Note: Removed artificial 0.0 filtering to accurately show crashes to zero

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
            # Fix scaling bug: Gold is logged as 0-100, others are 0-1.
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

def plot_llm_weights(run_dir, out_dir, smooth=10):
    weight_keys = ['w_wood', 'w_stone', 'w_workbench', 'w_iron', 'w_bridge', 'w_enemy', 'w_gold']
    
    plt.figure(figsize=(10, 6))
    
    has_data = False
    for key in weight_keys:
        full_tag = f"LLM_Weights_Softmax/{key}"
        steps, vals = get_scalars(run_dir, full_tag)
        if steps:
            has_data = True
            if smooth > 1:
                plt.plot(steps[smooth-1:], moving_average(vals, smooth), label=key, linewidth=2)
            else:
                plt.plot(steps, vals, label=key, linewidth=2)
                
    if has_data:
        plt.xlabel("Environment Steps")
        plt.ylabel("Weight Value")
        plt.title("LLM Softmax Attention Budget Over Time")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "LLM_Weights_Softmax.png"))
        plt.close()
        print("Saved LLM_Weights_Softmax.png")

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
    plot_llm_weights(run_dir, out_dir, smooth=10)
    
    # If using success rate logging
    plot_metric(run_dir, out_dir, 'Episodes/Success_Rate', "Success Rate", "Gold Mining Success Rate", "Episodes_Success_Rate.png", smooth=10)
