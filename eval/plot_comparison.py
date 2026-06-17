"""
plot_comparison.py — Generate Plots for Thesis Evaluation
==========================================================

Generates comparison plots from tensorboard logs.

Usage:
    python -m eval.plot_comparison
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from eval.sample_efficiency import load_run_data

def plot_success_rates():
    run_dirs = sorted(glob.glob("runs/v3_*"), key=os.path.getmtime)
    if not run_dirs:
        print("No runs found.")
        return
        
    plt.figure(figsize=(10, 6))
    
    for rd in run_dirs:
        name = os.path.basename(rd)
        data = load_run_data(rd)
        if "Episodes/Success_Rate" in data:
            steps = data["Episodes/Success_Rate"]["steps"]
            vals = data["Episodes/Success_Rate"]["values"]
            
            # Simple EMA smoothing
            smoothed = []
            ema = vals[0] if vals else 0
            for v in vals:
                ema = 0.9 * ema + 0.1 * v
                smoothed.append(ema)
                
            plt.plot(steps, smoothed, label=name, alpha=0.8)
            
    plt.title("Success Rate Comparison (v3)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Success Rate (Gold Mined)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("success_rate_comparison.png", dpi=300)
    print("Saved plot to success_rate_comparison.png")

if __name__ == "__main__":
    plot_success_rates()
