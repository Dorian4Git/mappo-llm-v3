import os
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

def plot_comparison(main_run1, label1, main_run2, label2, subtask_run1, subtask_run2, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    main_metrics = {
        "Rewards/Avg_Env_Reward": "Average Extrinsic Reward",
        "Episodes/Success_Rate": "Gold Mining Success Rate",
        "TD_Error/Abs_Mean": "Absolute Mean TD Error",
    }
    
    subtask_metrics = {
        "Subtasks/Pickaxe_Pct": "Pickaxe Subtask Completion Rate",
        "Subtasks/Iron_Pct": "Iron Subtask Completion Rate",
        "Subtasks/Sword_Pct": "Sword Subtask Completion Rate",
    }
    
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
            
            if "Rate" in title or "Pct" in title:
                plt.ylim(0, 1.05)
                
            plt.legend(loc='best')
            plt.tight_layout()
            
            safe_name = tag.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"Saved {out_path}")

    plot_set(main_metrics, main_run1, main_run2)
    plot_set(subtask_metrics, subtask_run1, subtask_run2)

if __name__ == "__main__":
    runs_dir = "C:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3/runs"
    
    main_lora = os.path.join(runs_dir, "v3_HRL_Std_E128_s42_20260619-202112")
    main_base = os.path.join(runs_dir, "v3_HRL_Std_NoLoRA_E128_s42_20260620-092622")
    
    subtask_lora = os.path.join(runs_dir, "v3_HRL_Std_LoRA_E128_s42_20260620-121351")
    subtask_base = os.path.join(runs_dir, "v3_HRL_Std_NoLoRA_E128_s42_20260620-135743")
    
    out_dir = r"C:\Users\doria\OneDrive\Documents\school\master IS\Semestre 4\Travail de Master\rev_v3\20260620\plots"
    
    print("Generating comparative plots...")
    plot_comparison(main_lora, "QLoRA Fine-tuned", main_base, "Base (No LoRA)", subtask_lora, subtask_base, out_dir)
    print("Done!")

