"""
critic_trigger.py — "Stuck and Confused" Two-Stage Critic Trigger
==================================================================

Implements the combined plateau + z-score trigger mechanism:

Stage 1 (The Gate — Plateau Detection):
    Check if the rolling average of success rate has stagnated
    over the last Y updates.

Stage 2 (The Trigger — Z-Score):
    If the gate is open (agents are stuck), check if the TD error
    variance has spiked above N standard deviations from the running mean.

This design enables the thesis to compare how different network depths
(2-layer vs 3-layer critic) alter the TD error landscape and trigger frequency.

Usage:
    trigger = CriticTrigger(config)
    # Register as a callback in the training loop:
    train_mappo_v3(callbacks=[trigger.on_update_end])
"""

import json
import os
import time
import numpy as np
from collections import deque
from typing import Optional


class CriticTrigger:
    """
    Two-stage "Stuck and Confused" LLM intervention trigger.

    Stage 1 — Plateau Gate:
        Monitors the rolling average of success rate (or env reward).
        The gate opens when improvement has stalled for `plateau_window` updates.

    Stage 2 — Z-Score Trigger:
        When the gate is open, monitors TD error variance.
        Fires when variance exceeds `sigma` standard deviations from the running mean.

    The trigger also tracks per-subtask plateaus for more targeted interventions.
    """

    def __init__(
        self,
        # Stage 1: Plateau Gate
        plateau_window: int = 10,
        plateau_threshold: float = 0.02,
        # Stage 2: Z-Score Trigger
        td_window_size: int = 100,
        td_sigma: float = 1.5,
        # Cooldown
        cooldown_updates: int = 5,
        # Subtask plateau
        subtask_plateau_updates: int = 20,
        # Sensitivity decay
        sensitivity_decay_rate: float = 0.995,
        # Orchestrator for LLM queries
        orchestrator=None,
        prompt_builder=None,
        reward_injector=None,
        # Logging
        log_dir: str = "data/interventions",
    ):
        # Stage 1 config
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold

        # Stage 2 config
        self.td_window_size = td_window_size
        self.td_sigma = td_sigma

        # Cooldown
        self.cooldown_updates = cooldown_updates

        # Subtask plateau
        self.subtask_plateau_updates = subtask_plateau_updates

        # Sensitivity decay
        self.sensitivity_decay_rate = sensitivity_decay_rate
        self._sensitivity = 1.0

        # External modules
        self.orchestrator = orchestrator
        self.prompt_builder = prompt_builder
        self.reward_injector = reward_injector

        # ── Internal State ───────────────────────────────────────────
        # Success rate history for plateau detection
        self._success_rate_history: deque = deque(maxlen=plateau_window * 2)
        self._env_reward_history: deque = deque(maxlen=plateau_window * 2)

        # TD error history for z-score computation
        self._td_mean_history: deque = deque(maxlen=td_window_size)
        self._td_var_history: deque = deque(maxlen=td_window_size)
        self._td_abs_mean_history: deque = deque(maxlen=td_window_size)

        # Per-subtask completion histories
        self._subtask_histories: dict[str, deque] = {}

        # Cooldown tracker
        self._last_trigger_update = -cooldown_updates  # Allow immediate first trigger
        self._total_triggers = 0

        # Logging
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._log_path = os.path.join(log_dir, f"trigger_log_{timestamp}.jsonl")
        self._log_file = open(self._log_path, "a", encoding="utf-8")

    def on_update_end(self, update: int, metrics: dict, td_stats: dict) -> Optional[dict]:
        """
        Callback fired after each training update.

        Implements the two-stage trigger logic and fires LLM intervention
        if conditions are met.

        Args:
            update: Current training update number.
            metrics: Dict with success_rate, avg_env_reward, subtask_pcts, etc.
            td_stats: Dict with mean_td_error, std_td_error, variance_td_error, etc.

        Returns:
            Optional dict with 'adaptive_weights' to override current weights,
            or None if no intervention triggered.
        """
        # ── Record histories ─────────────────────────────────────────
        success_rate = metrics.get("success_rate", 0.0)
        avg_env_reward = metrics.get("avg_env_reward", 0.0)
        subtask_pcts = metrics.get("subtask_pcts", {})

        self._success_rate_history.append(success_rate)
        self._env_reward_history.append(avg_env_reward)
        self._td_mean_history.append(td_stats.get("mean_td_error", 0.0))
        self._td_var_history.append(td_stats.get("variance_td_error", 0.0))
        self._td_abs_mean_history.append(td_stats.get("abs_mean_td_error", 0.0))

        for subtask, pct in subtask_pcts.items():
            if subtask not in self._subtask_histories:
                self._subtask_histories[subtask] = deque(maxlen=self.subtask_plateau_updates * 2)
            self._subtask_histories[subtask].append(pct)

        # ── Check trigger conditions ─────────────────────────────────
        should_trigger, trigger_reason = self._evaluate_trigger(update, metrics, td_stats)

        if not should_trigger:
            return None

        # ── Fire intervention ────────────────────────────────────────
        self._last_trigger_update = update
        self._total_triggers += 1
        self._sensitivity *= self.sensitivity_decay_rate

        print(f"\n[CriticTrigger] 🚨 INTERVENTION #{self._total_triggers} "
              f"at update {update}")
        print(f"      Reason: {trigger_reason}")

        # Build context for LLM
        intervention_result = self._execute_intervention(update, metrics, td_stats, trigger_reason)

        # Log the trigger event
        self._log_trigger(update, metrics, td_stats, trigger_reason, intervention_result)

        return intervention_result

    def _evaluate_trigger(
        self, update: int, metrics: dict, td_stats: dict
    ) -> tuple[bool, str]:
        """
        Evaluate the two-stage trigger.

        Returns:
            (should_trigger, reason_string)
        """
        # ── Cooldown check ───────────────────────────────────────────
        if update - self._last_trigger_update < self.cooldown_updates:
            return False, ""

        # Need enough history
        if len(self._success_rate_history) < self.plateau_window:
            return False, ""

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: Plateau Gate — Is the agent stuck?
        # ══════════════════════════════════════════════════════════════
        gate_open = False
        gate_reason = ""

        # Check success rate plateau
        recent = list(self._success_rate_history)[-self.plateau_window:]
        older = list(self._success_rate_history)[-self.plateau_window * 2:-self.plateau_window]

        if len(older) >= self.plateau_window // 2:
            recent_mean = np.mean(recent)
            older_mean = np.mean(older) if older else recent_mean
            improvement = recent_mean - older_mean

            if abs(improvement) < self.plateau_threshold:
                gate_open = True
                gate_reason = (
                    f"Success rate plateau: {older_mean:.3f} → {recent_mean:.3f} "
                    f"(Δ={improvement:+.4f}, threshold={self.plateau_threshold})"
                )

        # Also check if specific subtasks are stuck
        stuck_subtask = self._find_stuck_subtask()
        if stuck_subtask:
            gate_open = True
            gate_reason = (
                f"Subtask '{stuck_subtask}' plateau "
                f"(no improvement in {self.subtask_plateau_updates} updates)"
            )

        if not gate_open:
            return False, ""

        # ══════════════════════════════════════════════════════════════
        # STAGE 2: Z-Score Trigger — Is the critic confused?
        # ══════════════════════════════════════════════════════════════
        if len(self._td_var_history) < 10:
            # Not enough TD data yet — gate is open but we can't assess confusion
            # Trigger anyway with a weaker signal
            return True, f"[Gate Open] {gate_reason} (TD data insufficient, triggering on plateau alone)"

        td_vars = np.array(list(self._td_var_history))
        current_var = td_stats.get("variance_td_error", 0.0)
        td_var_mean = td_vars.mean()
        td_var_std = td_vars.std()

        if td_var_std > 1e-8:
            z_score = (current_var - td_var_mean) / td_var_std
        else:
            z_score = 0.0

        # Apply sensitivity decay (higher threshold as training progresses)
        effective_sigma = self.td_sigma / max(self._sensitivity, 0.1)

        if z_score > effective_sigma:
            reason = (
                f"[Gate+Trigger] {gate_reason} | "
                f"TD variance z-score={z_score:.2f} > σ={effective_sigma:.2f} "
                f"(var={current_var:.4f}, mean={td_var_mean:.4f}, std={td_var_std:.4f})"
            )
            return True, reason

        # Gate open but critic isn't confused enough — still trigger if plateau is severe
        if len(recent) >= self.plateau_window:
            recent_mean = np.mean(recent)
            if recent_mean < 0.05:  # Less than 5% success rate — definitely stuck
                return True, f"[Gate Open, Severe] {gate_reason} (success rate < 5%)"

        return False, ""

    def _find_stuck_subtask(self) -> Optional[str]:
        """
        Check if any subtask has plateaued for too long.

        Returns the name of the stuck subtask, or None.
        """
        for name, history in self._subtask_histories.items():
            if name in ("gameover",):
                continue
            if len(history) < self.subtask_plateau_updates:
                continue

            recent = list(history)[-self.subtask_plateau_updates:]
            recent_arr = np.array(recent)

            # A subtask is stuck if:
            # 1. It's not at 100% (not mastered)
            # 2. The variance over the window is very low (no improvement)
            mean_pct = recent_arr.mean()
            if 0.05 < mean_pct < 0.95 and recent_arr.std() < 0.02:
                return name

        return None

    def _execute_intervention(
        self, update: int, metrics: dict, td_stats: dict, reason: str
    ) -> Optional[dict]:
        """
        Execute the LLM intervention.

        Builds a prompt, queries the LLM, and returns new weights.
        """
        if self.orchestrator is None or self.prompt_builder is None:
            print("      [CriticTrigger] No orchestrator/prompt_builder configured — "
                  "returning default intervention")
            return None

        # Build the intervention prompt
        prompt = self.prompt_builder.build_intervention_prompt(
            metrics=metrics,
            td_stats=td_stats,
            trigger_reason=reason,
            trigger_count=self._total_triggers,
        )

        # Query LLM
        response = self.orchestrator.query_intervention(prompt)
        if response is None:
            print("      [CriticTrigger] LLM returned None — no intervention applied")
            return None

        # Process through reward injector
        if self.reward_injector:
            new_weights = self.reward_injector.process_intervention(
                response=response,
                current_weights=metrics.get("adaptive_weights", {}),
                metrics=metrics,
                update=update,
            )
            if new_weights:
                return {"adaptive_weights": new_weights}

        return None

    def _log_trigger(
        self, update: int, metrics: dict, td_stats: dict,
        reason: str, result: Optional[dict]
    ):
        """Log the trigger event to disk."""
        log_entry = {
            "update": update,
            "trigger_number": self._total_triggers,
            "reason": reason,
            "success_rate": metrics.get("success_rate", 0.0),
            "avg_env_reward": metrics.get("avg_env_reward", 0.0),
            "td_stats": td_stats,
            "subtask_pcts": metrics.get("subtask_pcts", {}),
            "sensitivity": self._sensitivity,
            "intervention_applied": result is not None,
            "new_weights": result.get("adaptive_weights", {}) if result else {},
        }
        self._log_file.write(json.dumps(log_entry) + "\n")
        self._log_file.flush()

    def get_stats(self) -> dict:
        """Return trigger statistics for evaluation."""
        return {
            "total_triggers": self._total_triggers,
            "current_sensitivity": self._sensitivity,
            "success_rate_trend": list(self._success_rate_history)[-20:],
            "td_variance_trend": list(self._td_var_history)[-20:],
        }

    def close(self):
        """Close log file."""
        self._log_file.close()
