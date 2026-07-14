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
    parser.add_argument("--fair-mode", action="store_true",
                        help="Disable programmatic DAG guardrails and enable softmax attention budget")
    parser.add_argument("--softmax-temp", type=float, default=0.5,
                        help="Temperature for softmax attention budget (lower = sharper, default 0.5)")
    args = parser.parse_args()

    import yaml
    import os

    # Load the config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "llm_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    llm_cfg = config.get("llm", {})

    # Override argparse defaults if config is present
    backend = args.llm_backend if args.llm_backend != "ollama" else llm_cfg.get("backend", "ollama")
    model_name = args.llm_model if args.llm_model != "qwen2.5:7b" else llm_cfg.get("model_name", "qwen2.5:7b")
    rate_limit_rpm = llm_cfg.get("rate_limit_rpm")

    # Set up the LLM pipeline
    bridge = LLMBridge(
        backend=backend, 
        model_name=model_name,
        rate_limit_rpm=rate_limit_rpm
    )
    orchestrator = LLMOrchestratorV2(
        model_name=model_name, 
        bridge=bridge,
        enforce_dag_guardrails=not args.fair_mode
    )
    prompt_builder = PromptBuilder()
    reward_injector = RewardInjector(
        enforce_dag_guardrails=not args.fair_mode,
        use_softmax_attention=args.fair_mode,
        softmax_temperature=args.softmax_temp,
    )

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
        llm_model_name=model_name,
        fair_mode=args.fair_mode,
        orchestrator=orchestrator,
    )

    # Cleanup
    critic_trigger.close()
    bridge.close()
    if traj_logger:
        traj_logger.close()
