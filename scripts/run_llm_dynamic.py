"""
run_llm_dynamic.py — Launch Critic-Triggered LLM Training
===========================================================
Runs the full MAPPO training with the two-stage critic trigger
and LLM adaptive weight adjustment.

Usage:
    python scripts/run_llm_dynamic.py
    python scripts/run_llm_dynamic.py --deep  # 3-layer critic
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train_loop import train_mappo_v3
from llm.async_bridge import LLMBridge
from llm.orchestrator import LLMOrchestratorV2
from llm.critic_trigger import CriticTrigger
from llm.prompt_builder import PromptBuilder
from llm.reward_injector import RewardInjector

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Critic-Triggered LLM Training")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--llm-interval", type=int, default=10,
                        help="Legacy LLM query interval (also enables periodic weight queries)")
    parser.add_argument("--clear-logs", action="store_true")
    parser.add_argument("--clear-checkpoints", action="store_true")
    parser.add_argument("--enable-logging", action="store_true")
    parser.add_argument("--critic-trigger-only", action="store_true",
                        help="Use only critic trigger (no periodic LLM queries)")
    parser.add_argument("--llm-backend", type=str, default="ollama", choices=["ollama", "huggingface", "gemini"])
    parser.add_argument("--llm-model", type=str, default="qwen2.5:7b")
    args = parser.parse_args()

    # Set up the LLM pipeline
    bridge = LLMBridge(backend=args.llm_backend, model_name=args.llm_model)
    orchestrator = LLMOrchestratorV2(model_name=args.llm_model, bridge=bridge)
    prompt_builder = PromptBuilder()
    reward_injector = RewardInjector()

    critic_trigger = CriticTrigger(
        orchestrator=orchestrator,
        prompt_builder=prompt_builder,
        reward_injector=reward_injector,
    )

    # Trajectory logging
    traj_logger = None
    if args.enable_logging:
        from logging_utils.trajectory_logger import TrajectoryLogger
        traj_logger = TrajectoryLogger(output_dir="data/trajectories")

    # Callbacks
    callbacks = [critic_trigger.on_update_end]

    train_mappo_v3(
        clear_logs=args.clear_logs,
        clear_checkpoints=args.clear_checkpoints,
        n_envs=args.n_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        no_shaping=False,
        llm_dynamic=not args.critic_trigger_only,
        llm_interval=args.llm_interval,
        deep=args.deep,
        seed=args.seed,
        callbacks=callbacks,
        trajectory_logger=traj_logger,
        llm_model_name=args.llm_model,
    )

    # Cleanup
    critic_trigger.close()
    bridge.close()
    if traj_logger:
        traj_logger.close()
