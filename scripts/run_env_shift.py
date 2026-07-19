"""
run_env_shift.py — Launch Environment Shift Experiment
========================================================
Runs the LLM dynamic MAPPO training with a mid-training semantic
rename of game zones (e.g., "Workbench" → "Furnace" at epoch 500).

This proves zero-shot semantic adaptation: the LLM's latent world model
understands that a "Furnace" serves a similar crafting function to a
"Workbench" without any rule updates or retraining.

Usage:
    # Main experiment: Workbench → Furnace
    python scripts/run_env_shift.py --shift-name Furnace

    # Nonsense control: Workbench → Zylorg
    python scripts/run_env_shift.py --shift-name Zylorg

    # Quick smoke test
    python scripts/run_env_shift.py --shift-name Furnace --shift-update 3 --num-updates 5 --n-envs 8
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
    parser = argparse.ArgumentParser(description="Run Environment Shift Experiment")
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-interval", type=int, default=10)
    parser.add_argument("--llm-backend", type=str, default="ollama", choices=["ollama", "huggingface", "gemini"])
    parser.add_argument("--llm-model", type=str, default="qwen2.5:7b")
    parser.add_argument("--fair-mode", action="store_true",
                        help="Disable DAG guardrails + enable softmax attention budget")
    parser.add_argument("--softmax-temp", type=float, default=0.5)
    parser.add_argument("--clear-logs", action="store_true")
    parser.add_argument("--clear-checkpoints", action="store_true")
    parser.add_argument("--enable-logging", action="store_true")
    # Environment shift params
    parser.add_argument("--shift-name", type=str, required=True,
                        help="New name for Workbench zone (e.g., 'Furnace' or 'Zylorg')")
    parser.add_argument("--shift-update", type=int, default=500,
                        help="Epoch at which to activate the environment shift (default: 500)")
    args = parser.parse_args()

    import yaml

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "llm_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    llm_cfg = config.get("llm", {})
    backend = args.llm_backend if args.llm_backend != "ollama" else llm_cfg.get("backend", "ollama")
    model_name = args.llm_model if args.llm_model != "qwen2.5:7b" else llm_cfg.get("model_name", "qwen2.5:7b")
    rate_limit_rpm = llm_cfg.get("rate_limit_rpm")

    # Zone aliases (applied at shift_update epoch)
    zone_aliases = {"workbench": args.shift_name}

    # Set up pipeline (aliases NOT active yet — they activate at shift_update)
    bridge = LLMBridge(
        backend=backend,
        model_name=model_name,
        rate_limit_rpm=rate_limit_rpm,
    )
    orchestrator = LLMOrchestratorV2(
        model_name=model_name,
        bridge=bridge,
        enforce_dag_guardrails=not args.fair_mode,
        # zone_aliases NOT set here — set mid-training by train_loop
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

    traj_logger = None
    if args.enable_logging:
        from logging_utils.trajectory_logger import TrajectoryLogger
        traj_logger = TrajectoryLogger(output_dir="data/trajectories")

    callbacks = [critic_trigger.on_update_end]

    safe_model = model_name.replace(":", "-").replace("/", "-")
    safe_shift = args.shift_name.replace(" ", "_")
    run_prefix = f"v3_EnvShift_{safe_shift}_{safe_model}"

    print(f"[EnvShift] Environment Shift experiment with {model_name}")
    print(f"[EnvShift] Zone rename: Workbench -> {args.shift_name} at epoch {args.shift_update}")
    print(f"[EnvShift] Run prefix: {run_prefix}")

    train_mappo_v3(
        clear_logs=args.clear_logs,
        clear_checkpoints=args.clear_checkpoints,
        n_envs=args.n_envs,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        no_shaping=False,
        llm_dynamic=True,
        llm_interval=args.llm_interval,
        seed=args.seed,
        callbacks=callbacks,
        trajectory_logger=traj_logger,
        llm_model_name=model_name,
        fair_mode=args.fair_mode,
        orchestrator=orchestrator,
        zone_aliases=zone_aliases,
        env_shift_update=args.shift_update,
        run_prefix_override=run_prefix,
    )

    critic_trigger.close()
    bridge.close()
    if traj_logger:
        traj_logger.close()
