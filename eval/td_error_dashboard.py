"""
td_error_dashboard.py — Analyze TD Errors
===========================================
Generates visualizations of critic confusion via TD error magnitude.
"""
import os
import glob
import matplotlib.pyplot as plt
from eval.sample_efficiency import load_run_data

def analyze_td_error():
    run_dirs = sorted(glob.glob("runs/v3_*"), key=os.path.getmtime)
    if not run_dirs:
        print("No tensorboard runs found.")
        return
        
    plt.figure(figsize=(10, 6))
    
    for rd in run_dirs:
        name = os.path.basename(rd)
        data = load_run_data(rd)
        
        if "TD_Error/Abs_Mean" in data:
            steps = data["TD_Error/Abs_Mean"]["steps"]
            vals = data["TD_Error/Abs_Mean"]["values"]
            
            # EMA Smoothing to match training visualization
            smoothed = []
            ema = vals[0] if vals else 0
            for v in vals:
                ema = 0.9 * ema + 0.1 * v
                smoothed.append(ema)
                
            plt.plot(steps, smoothed, label=f"{name}", linewidth=2, alpha=0.8)
            
    plt.title("Critic Confusion: Absolute Mean TD Error over Training", pad=15, fontweight="bold")
    plt.xlabel("Environment Steps")
    plt.ylabel("Absolute Mean TD Error")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    out_path = "plots/td_error_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved TD error plot to {out_path}")

if __name__ == "__main__":
    analyze_td_error()
