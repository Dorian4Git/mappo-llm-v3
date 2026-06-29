import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_logs(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the tfevents file
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        print(f"No event files found in {log_dir}")
        return
        
    path = os.path.join(log_dir, event_files[0])
    print(f"Loading {path}...")
    
    # Load the event accumulator
    ea = EventAccumulator(path)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"Found scalar tags: {tags}")
    
    # Base metrics
    metrics = {
        "Rewards/Avg_Env_Reward": "Average Extrinsic Reward",
        "Episodes/Success_Rate": "Gold Mining Success Rate",
        "TD_Error/Abs_Mean": "Absolute Mean TD Error"
    }
    
    # Dynamically find all subtasks
    for tag in tags:
        if "Subtasks" in tag:
            metrics[tag] = tag.split("/")[-1].replace("_Pct", " Subtask Completion Rate")
            
    for tag, title in metrics.items():
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, linewidth=2.5, color='#1f77b4')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Format the plot
            plt.title(f"{title} over Training Steps", pad=15, fontweight="bold")
            plt.xlabel("Global Step")
            plt.ylabel(title)
            
            if "Success_Rate" in tag or "Rate" in title:
                plt.ylim(0, 1.05)
                
            plt.tight_layout()
            
            # Save the plot
            safe_name = tag.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Tensorboard logs for a single run.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the run directory containing tfevents.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the plots.")
    args = parser.parse_args()
    
    plot_tensorboard_logs(args.log_dir, args.output_dir)
