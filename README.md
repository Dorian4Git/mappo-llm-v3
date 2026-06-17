# MAPPO-LLM-V3

## Overview
This project extends MAPPO-LLM-V2 with:
1. **Critic-Triggered LLM Interventions:** Replaces periodic queries with a two-stage "Stuck and Confused" trigger based on success plateaus and TD error variance.
2. **QLoRA Fine-Tuning:** Pipeline to generate bottleneck→resolution datasets from successful trajectories, and fine-tune a specialized LLM expert.
3. **HRL Options Framework:** Extends shaping to high-level options with intrinsic rewards.

## Phases
* **Phase 1:** Core RL, Trajectory Logging, Async Bridge
* **Phase 2:** Critic-Triggered Interventions, Prompt Builder, Reward Injector
* **Phase 3:** QLoRA Fine-Tuning Pipeline
* **Phase 4:** HRL Options Framework
* **Phase 5:** Evaluation & Visualization

## Quickstart

Run a baseline without LLM shaping (to collect trajectories):
```bash
python scripts/run_baseline.py --enable-logging

python scripts/run_baseline.py --enable-logging --static-shaping  # Fair comparison
```

Run LLM dynamic shaping (with standard 2-layer critic):
```bash
python scripts/run_llm_dynamic.py --enable-logging
```

Run the complete fine-tuning pipeline on collected trajectories:
```bash
python scripts/run_finetune.py
```
