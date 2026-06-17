"""
compare_architectures.py — Evaluate 2-Layer vs 3-Layer Critic Trigger
========================================================================

Parses tensorboard logs and critic trigger logs to compare how network
architecture impacts the "Stuck and Confused" LLM interventions.

Usage:
    python -m eval.compare_architectures
"""

import json
import os
import glob
import numpy as np

def analyze_trigger_logs(log_dir: str = "data/interventions"):
    """Analyze trigger behavior based on JSONL logs."""
    logs = glob.glob(os.path.join(log_dir, "trigger_log_*.jsonl"))
    if not logs:
        print(f"No trigger logs found in {log_dir}")
        return

    print("=== Critic Trigger Analysis ===")
    
    for log_file in logs:
        triggers = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    triggers.append(json.loads(line))
                    
        if not triggers:
            continue
            
        print(f"\nLog: {os.path.basename(log_file)}")
        print(f"Total interventions: {len(triggers)}")
        
        # Breakdown by reason
        reasons = {}
        for t in triggers:
            r = t.get("reason", "Unknown")
            if "Gate+Trigger" in r:
                r_type = "Plateau + Variance Spike"
            elif "Gate Open, Severe" in r:
                r_type = "Severe Plateau"
            else:
                r_type = "Plateau Only"
            reasons[r_type] = reasons.get(r_type, 0) + 1
            
        for k, v in reasons.items():
            print(f"  - {k}: {v} ({v/len(triggers):.0%})")
            
        # Average TD Variance when triggered
        avg_var = np.mean([t["td_stats"].get("variance_td_error", 0) for t in triggers])
        print(f"Average TD Variance at trigger: {avg_var:.6f}")

if __name__ == "__main__":
    analyze_trigger_logs()
