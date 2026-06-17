"""
run_baseline.py — Launch Baseline Training
============================================
Convenience launcher for sparse baseline (no LLM shaping).

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --num-updates 10 --n-envs 8  # quick test
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train_loop import train_mappo_v3

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Baseline Training")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--clear-logs", action="store_true")
    parser.add_argument("--clear-checkpoints", action="store_true")
    parser.add_argument("--enable-logging", action="store_true")
    parser.add_argument("--static-shaping", action="store_true", help="Enable static potential-based reward shaping (fair comparison)")
    args = parser.parse_args()

    traj_logger = None
    if args.enable_logging:
        from logging_utils.trajectory_logger import TrajectoryLogger
        traj_logger = TrajectoryLogger(output_dir="data/trajectories")

    train_mappo_v3(
        clear_logs=args.clear_logs,
        clear_checkpoints=args.clear_checkpoints,
        n_envs=args.n_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        no_shaping=not args.static_shaping, # If static shaping is ON, no_shaping is FALSE
        llm_dynamic=False,                  # Force LLM dynamic OFF for baseline
        deep=args.deep,
        seed=args.seed,
        trajectory_logger=traj_logger,
    )
