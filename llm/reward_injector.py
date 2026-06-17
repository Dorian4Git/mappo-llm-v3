"""
reward_injector.py — Safe Reward Shaping Injection from LLM Responses
======================================================================

Parses LLM-generated reward weight adjustments, validates against safety
bounds, applies EMA smoothing, and supports auto-rollback if performance drops.

Usage:
    injector = RewardInjector()
    new_weights = injector.process_intervention(response, current_weights, update)
"""

import json
import os
import time
import numpy as np
from collections import deque
from typing import Optional, Dict

from llm.orchestrator import apply_dag_guardrails


WEIGHT_KEYS = ['w_wood', 'w_stone', 'w_workbench', 'w_iron', 'w_bridge', 'w_enemy', 'w_gold']


class RewardInjector:
    """
    Safely injects LLM-generated reward shaping rules into the training loop.

    Safety features:
    - Weight clipping (per-weight max and total sum bound)
    - EMA smoothing for gradual transitions
    - Normalization (average weight stays near 1.0)
    - Auto-rollback if performance drops significantly after an intervention
    """

    def __init__(
        self,
        max_weight: float = 3.0,
        weight_sum_bound: float = 15.0,
        ema_alpha: float = 0.2,
        rollback_enabled: bool = True,
        rollback_window: int = 25,
        rollback_drop_threshold: float = 0.20,
        log_dir: str = "data/interventions",
    ):
        self.max_weight = max_weight
        self.weight_sum_bound = weight_sum_bound
        self.ema_alpha = ema_alpha
        self.rollback_enabled = rollback_enabled
        self.rollback_window = rollback_window
        self.rollback_drop_threshold = rollback_drop_threshold

        # Rollback state
        self._pre_intervention_weights: Optional[dict] = None
        self._pre_intervention_reward: Optional[float] = None
        self._intervention_update: Optional[int] = None
        self._rollback_history: list[dict] = []

        # Post-intervention reward tracking
        self._post_intervention_rewards: deque = deque(maxlen=rollback_window)

        # Logging
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._log_path = os.path.join(log_dir, f"injector_log_{timestamp}.jsonl")
        self._log_file = open(self._log_path, "a", encoding="utf-8")

    def process_intervention(
        self,
        response: dict,
        current_weights: dict,
        metrics: dict,
        update: int,
    ) -> Optional[dict]:
        """
        Process an LLM intervention response and return safe new weights.

        Args:
            response: Parsed JSON dict from the LLM.
            current_weights: Current adaptive weights dict.
            metrics: Dict containing current task completion percentages.
            update: Current training update number.

        Returns:
            New weights dict, or None if the response was invalid.
        """
        if response is None:
            return None

        reasoning = response.get("reasoning", "No reasoning provided")
        print(f"      [Injector] LLM reasoning: {reasoning}")

        # ── Parse raw weights from response ──────────────────────────
        raw_weights = {}
        for k in WEIGHT_KEYS:
            if k in response:
                try:
                    raw_weights[k] = float(response[k])
                except (ValueError, TypeError):
                    raw_weights[k] = current_weights.get(k, 1.0)
            else:
                raw_weights[k] = current_weights.get(k, 1.0)

        # ── Safety: Clip individual weights ──────────────────────────
        clipped = {k: float(np.clip(v, 0.0, self.max_weight)) for k, v in raw_weights.items()}

        # ── Safety: Bound total weight sum ───────────────────────────
        total = sum(clipped.values())
        if total > self.weight_sum_bound:
            scale = self.weight_sum_bound / total
            clipped = {k: v * scale for k, v in clipped.items()}

        # ── EMA Smoothing ────────────────────────────────────────────
        alpha = self.ema_alpha
        smoothed = {}
        for k in WEIGHT_KEYS:
            old = current_weights.get(k, 1.0)
            smoothed[k] = alpha * clipped[k] + (1.0 - alpha) * old

        # ── Programmatic Guardrails to strictly enforce DAG impossibilities
        smoothed = apply_dag_guardrails(smoothed, metrics)

        # ── Normalization (average ≈ 1.0) ────────────────────────────
        w_sum = sum(smoothed.values()) + 1e-6
        factor = len(WEIGHT_KEYS) / w_sum
        normalized = {k: v * factor for k, v in smoothed.items()}

        # ── Store rollback state ─────────────────────────────────────
        if self.rollback_enabled:
            self._pre_intervention_weights = dict(current_weights)
            self._intervention_update = update
            self._post_intervention_rewards.clear()

        # ── Log the injection ────────────────────────────────────────
        log_entry = {
            "update": update,
            "reasoning": reasoning,
            "raw_weights": raw_weights,
            "clipped_weights": clipped,
            "smoothed_weights": smoothed,
            "final_weights": normalized,
            "previous_weights": current_weights,
        }
        self._log_file.write(json.dumps(log_entry) + "\n")
        self._log_file.flush()

        w_str = ", ".join(f"{k.replace('w_', '')}={v:.3f}" for k, v in normalized.items())
        print(f"      [Injector] New weights: {w_str}")

        return normalized

    def check_rollback(self, update: int, avg_env_reward: float, current_weights: dict) -> Optional[dict]:
        """
        Check if the last intervention caused a performance drop and should be rolled back.

        Call this from the training loop after each update.

        Args:
            update: Current update.
            avg_env_reward: Current environment reward.
            current_weights: Current weights (to compare against pre-intervention).

        Returns:
            Pre-intervention weights if rollback triggered, None otherwise.
        """
        if not self.rollback_enabled:
            return None
        if self._intervention_update is None:
            return None
        if self._pre_intervention_weights is None:
            return None

        # Track post-intervention rewards
        self._post_intervention_rewards.append(avg_env_reward)

        # Wait for the rollback window to fill
        updates_since = update - self._intervention_update
        if updates_since < self.rollback_window:
            return None

        # Compute average post-intervention reward
        if self._pre_intervention_reward is not None:
            post_avg = np.mean(list(self._post_intervention_rewards))
            pre_reward = self._pre_intervention_reward
            drop = (pre_reward - post_avg) / max(abs(pre_reward), 1e-6)

            if drop > self.rollback_drop_threshold:
                print(f"\n      [Injector] ⚠️ ROLLBACK triggered at update {update}!")
                print(f"      Pre-intervention reward: {pre_reward:.4f}")
                print(f"      Post-intervention avg:   {post_avg:.4f}")
                print(f"      Drop: {drop:.1%} > threshold {self.rollback_drop_threshold:.1%}")

                self._rollback_history.append({
                    "update": update,
                    "pre_reward": pre_reward,
                    "post_avg": post_avg,
                    "drop": drop,
                })

                old_weights = dict(self._pre_intervention_weights)
                self._pre_intervention_weights = None
                self._intervention_update = None
                return old_weights

        # Clear the rollback check (intervention was okay)
        self._pre_intervention_reward = avg_env_reward
        self._pre_intervention_weights = None
        self._intervention_update = None
        return None

    def close(self):
        """Close log file."""
        self._log_file.close()
