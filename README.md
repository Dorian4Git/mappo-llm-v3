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

python scripts/run_llm_dynamic.py --enable-logging --llm-backend gemini --llm-model gemini-2.5-flash

python scripts/run_llm_dynamic.py --enable-logging --llm-backend gemini --llm-model gemini-3.1-flash-lite
```

Run the complete fine-tuning pipeline on collected trajectories:
```bash
python scripts/run_finetune.py
```

Run HRL Options Framework (defaults to using the fine-tuned LoRA adapter):
```bash
python scripts/run_hrl.py --enable-logging
```

To benchmark the fine-tuned adapter against the base Qwen model, use the `--disable-lora` flag to dynamically run inference without the fine-tuned weights:
```bash
python scripts/run_hrl.py --enable-logging --disable-lora
```
