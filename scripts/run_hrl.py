"""
run_hrl.py — Launch HRL Options Training
==========================================
Placeholder for HRL training with Options framework.
To be fully integrated after Phase 2 & 3 validation.

Usage:
    python scripts/run_hrl.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from hrl.hrl_train_loop import train_mappo_hrl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HRL Options Training")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--enable-logging", action="store_true")
    parser.add_argument("--llm-backend", type=str, default="ollama", choices=["ollama", "huggingface", "gemini"])
    parser.add_argument("--llm-model", type=str, default="qwen2.5:7b")
    args = parser.parse_args()

    train_mappo_hrl(
        n_envs=args.n_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        deep=args.deep,
        seed=args.seed,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
    )
