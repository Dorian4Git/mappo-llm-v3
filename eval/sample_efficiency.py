"""
sample_efficiency.py — Compute Sample Efficiency Metrics
=========================================================

Analyzes TensorBoard logs to compute:
    - Environment steps to reach 50%/80%/95% success rate
    - Wall-clock time to convergence
    - LLM query count and total inference time
    - Comparison tables in LaTeX format

Usage:
    python -m eval.sample_efficiency
"""

import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run_data(run_dir: str) -> dict:
    """Load scalar data from a TensorBoard run directory."""
    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return {}

    event_files.sort(key=os.path.getmtime, reverse=True)
    ea = EventAccumulator(event_files[0])
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
            "wall_times": [e.wall_time for e in events],
        }
    return data


def find_threshold_step(data: dict, tag: str, threshold: float) -> dict:
    """
    Find the first step where a metric exceeds a threshold.

    Returns dict with step, wall_time, and value.
    """
    if tag not in data:
        return {"step": None, "wall_time": None, "value": None}

    steps = data[tag]["steps"]
    values = data[tag]["values"]
    wall_times = data[tag]["wall_times"]

    # Use EMA smoothing to avoid noise
    smoothed = []
    ema = values[0] if values else 0
    for v in values:
        ema = 0.9 * ema + 0.1 * v
        smoothed.append(ema)

    for i, sv in enumerate(smoothed):
        if sv >= threshold:
            return {
                "step": steps[i],
                "wall_time": wall_times[i] - wall_times[0] if wall_times else None,
                "value": sv,
            }

    return {"step": None, "wall_time": None, "value": max(smoothed) if smoothed else None}


def compute_metrics(run_dir: str, run_name: str) -> dict:
    """Compute all sample efficiency metrics for a single run."""
    data = load_run_data(run_dir)
    if not data:
        return {"name": run_name, "error": "No data found"}

    thresholds = [0.5, 0.8, 0.95]
    results = {"name": run_name}

    for t in thresholds:
        key = f"steps_to_{int(t*100)}pct"
        r = find_threshold_step(data, "Episodes/Success_Rate", t)
        results[key] = r["step"]
        results[f"time_to_{int(t*100)}pct_s"] = r["wall_time"]

    # Final performance
    if "Episodes/Success_Rate" in data:
        results["final_success_rate"] = data["Episodes/Success_Rate"]["values"][-1]
    if "Rewards/Avg_Env_Reward" in data:
        results["final_env_reward"] = data["Rewards/Avg_Env_Reward"]["values"][-1]

    # Total training steps
    if "Episodes/Success_Rate" in data:
        results["total_steps"] = data["Episodes/Success_Rate"]["steps"][-1]
        wall = data["Episodes/Success_Rate"]["wall_times"]
        results["total_wall_time_s"] = wall[-1] - wall[0] if wall else None

    return results


def format_latex_table(all_results: list[dict]) -> str:
    """Format results as a LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Sample Efficiency Comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Run & Steps→50\% & Steps→80\% & Steps→95\% & Final SR & Time (s) \\",
        r"\hline",
    ]

    for r in all_results:
        name = r.get("name", "?")
        s50 = r.get("steps_to_50pct", "—")
        s80 = r.get("steps_to_80pct", "—")
        s95 = r.get("steps_to_95pct", "—")
        fsr = r.get("final_success_rate", 0)
        tt = r.get("total_wall_time_s", 0)

        s50_str = f"{s50:,}" if isinstance(s50, int) else "—"
        s80_str = f"{s80:,}" if isinstance(s80, int) else "—"
        s95_str = f"{s95:,}" if isinstance(s95, int) else "—"
        fsr_str = f"{fsr:.1%}" if isinstance(fsr, (int, float)) else "—"
        tt_str = f"{tt:.0f}" if isinstance(tt, (int, float)) and tt else "—"

        lines.append(f"{name} & {s50_str} & {s80_str} & {s95_str} & {fsr_str} & {tt_str} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    # Auto-detect v3 runs
    run_dirs = sorted(glob.glob("runs/v3_*"), key=os.path.getmtime)

    if not run_dirs:
        print("No v3 runs found in runs/ directory.")
        return

    print(f"Found {len(run_dirs)} runs:\n")

    all_results = []
    for rd in run_dirs:
        name = os.path.basename(rd)
        print(f"  Analyzing: {name}")
        metrics = compute_metrics(rd, name)
        all_results.append(metrics)

        for key in ["steps_to_50pct", "steps_to_80pct", "steps_to_95pct",
                     "final_success_rate", "total_wall_time_s"]:
            val = metrics.get(key)
            if val is not None:
                if "steps" in key:
                    print(f"    {key}: {val:,}")
                elif "rate" in key:
                    print(f"    {key}: {val:.1%}")
                elif "time" in key:
                    print(f"    {key}: {val:.0f}s")
        print()

    # LaTeX table
    latex = format_latex_table(all_results)
    print("\n=== LaTeX Table ===\n")
    print(latex)

    # Save
    with open("sample_efficiency_results.tex", "w") as f:
        f.write(latex)
    print("\nSaved to sample_efficiency_results.tex")


if __name__ == "__main__":
    main()
