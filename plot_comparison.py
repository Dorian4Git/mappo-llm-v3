import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_events(log_dir, tag):
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        return [], []
    
    path = os.path.join(log_dir, event_files[0])
    ea = EventAccumulator(path)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    if tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return [], []

def get_all_tags(log_dir):
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        return []
    path = os.path.join(log_dir, event_files[0])
    ea = EventAccumulator(path)
    ea.Reload()
    return ea.Tags()['scalars']

def plot_comparison(main_run1, label1, main_run2, label2, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    main_metrics = {
        "Rewards/Avg_Env_Reward": "Average Extrinsic Reward",
        "Episodes/Success_Rate": "Gold Mining Success Rate",
        "TD_Error/Abs_Mean": "Absolute Mean TD Error",
    }
    
    # Discover subtasks from both runs
    tags1 = get_all_tags(main_run1)
    tags2 = get_all_tags(main_run2)
    all_tags = set(tags1) | set(tags2)
    
    subtask_metrics = {}
    for tag in all_tags:
        if "Subtasks" in tag:
            subtask_metrics[tag] = tag.split("/")[-1].replace("_Pct", " Subtask Completion Rate")
    
    def plot_set(metrics, r1, r2):
        for tag, title in metrics.items():
            steps1, values1 = load_events(r1, tag)
            steps2, values2 = load_events(r2, tag)
            
            if not steps1 and not steps2:
                print(f"Skipping {tag}, no data found in either run.")
                continue
                
            plt.figure(figsize=(10, 6))
            
            if steps1:
                plt.plot(steps1, values1, linewidth=2.5, label=label1, color='#1f77b4') # Blue
            if steps2:
                plt.plot(steps2, values2, linewidth=2.5, label=label2, color='#ff7f0e', alpha=0.8) # Orange
                
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title(f"{title} over Training Steps", pad=15, fontweight="bold")
            plt.xlabel("Global Step")
            plt.ylabel(title)
            
            if "Rate" in title or "Pct" in title or "Success_Rate" in tag:
                plt.ylim(0, 1.05)
                
            plt.legend(loc='best')
            plt.tight_layout()
            
            safe_name = tag.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")

    plot_set(main_metrics, main_run1, main_run2)
    plot_set(subtask_metrics, main_run1, main_run2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison between two Tensorboard runs.")
    parser.add_argument("--run1", type=str, required=True, help="Path to first run directory (e.g. LoRA)")
    parser.add_argument("--label1", type=str, required=True, help="Label for first run")
    parser.add_argument("--run2", type=str, required=True, help="Path to second run directory (e.g. Base)")
    parser.add_argument("--label2", type=str, required=True, help="Label for second run")
    parser.add_argument("--output_dir", type=str, default="plots/comparison", help="Directory to save the plots")
    args = parser.parse_args()
    
    print("Generating comparative plots...")
    plot_comparison(args.run1, args.label1, args.run2, args.label2, args.output_dir)
    print("Done!")

