import os
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
    
    # We want to plot: Rewards/Avg_Env_Reward, Episodes/Success_Rate, and TD_Error/Abs_Mean
    metrics = {
        "Rewards/Avg_Env_Reward": "Average Extrinsic Reward",
        "Episodes/Success_Rate": "Gold Mining Success Rate",
        "TD_Error/Abs_Mean": "Absolute Mean TD Error"
    }
    
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
            
            if "Success_Rate" in tag:
                plt.ylim(0, 1.05)
                
            plt.tight_layout()
            
            # Save the plot
            safe_name = tag.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")

if __name__ == "__main__":
    # Point to the last successful run
    log_directory = "runs/v3_HRL_Std_E128_s42_20260617-192630"
    output_directory = "plots"
    plot_tensorboard_logs(log_directory, output_directory)
