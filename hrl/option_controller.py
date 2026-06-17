"""
option_controller.py — High-Level Option Controller
=====================================================

Manages the active Option for each environment, handles option switching
via the critic trigger, and provides intrinsic rewards to the low-level MAPPO.

Usage:
    controller = OptionController(orchestrator, prompt_builder)
    # In training loop:
    intrinsic_r = controller.step(inventory, positions, zones)
    # On critic trigger:
    controller.request_new_options(metrics, td_stats)
"""

import numpy as np
from typing import Optional

from hrl.options import (
    Option, NUM_OPTIONS, OPTION_NAMES,
    can_initiate, is_terminated, compute_intrinsic_reward,
    option_to_one_hot, OPTION_TARGET_ZONE,
)


class OptionController:
    """
    High-level controller managing Options for each environment.

    Integrates with the critic trigger: when TD error triggers,
    instead of adjusting weights, the LLM assigns a new Option.
    """

    def __init__(
        self,
        n_envs: int,
        option_timeout: int = 200,
        intrinsic_scale: float = 0.1,
        orchestrator=None,
        prompt_builder=None,
    ):
        self.n_envs = n_envs
        self.option_timeout = option_timeout
        self.intrinsic_scale = intrinsic_scale
        self.orchestrator = orchestrator
        self.prompt_builder = prompt_builder

        # Per-environment state
        self._active_options = np.full(n_envs, Option.IDLE, dtype=np.int32)
        self._option_step_counts = np.zeros(n_envs, dtype=np.int32)
        self._option_history: list[list[dict]] = [[] for _ in range(n_envs)]

        # Statistics
        self._total_option_switches = 0
        self._option_success_counts = np.zeros(NUM_OPTIONS, dtype=np.int32)
        self._option_failure_counts = np.zeros(NUM_OPTIONS, dtype=np.int32)

    @property
    def active_options(self) -> np.ndarray:
        """[n_envs] array of current option indices."""
        return self._active_options

    def get_option_embeddings(self) -> np.ndarray:
        """
        Get one-hot encoded option embeddings for all environments.

        Returns:
            [n_envs, NUM_OPTIONS] float array.
        """
        return option_to_one_hot(self._active_options)

    def step(
        self,
        inventory: np.ndarray,
        positions: np.ndarray,
        zones: np.ndarray,
    ) -> np.ndarray:
        """
        Execute one step of the option controller.

        Checks for option termination/timeout and computes intrinsic rewards.

        Args:
            inventory: [n_envs, NUM_ITEMS]
            positions: [n_envs, 2, 2] agent positions
            zones: [n_envs, 7, 2] zone positions

        Returns:
            [n_envs, 2] intrinsic rewards
        """
        self._option_step_counts += 1

        # ── Check termination for each option ────────────────────────
        intrinsic_rewards = np.zeros((self.n_envs, 2), dtype=np.float32)

        for opt in Option:
            mask = (self._active_options == opt)
            if not mask.any():
                continue

            # Check if terminated (success)
            terminated = is_terminated(opt, inventory) & mask
            if terminated.any():
                self._option_success_counts[opt] += int(terminated.sum())
                # Assign next option based on DAG
                self._assign_dag_options(np.where(terminated)[0], inventory)

            # Check timeout (failure)
            timed_out = (self._option_step_counts >= self.option_timeout) & mask & ~terminated
            if timed_out.any():
                self._option_failure_counts[opt] += int(timed_out.sum())
                self._assign_dag_options(np.where(timed_out)[0], inventory)

            # Compute intrinsic reward for still-active envs
            still_active = mask & ~terminated & ~timed_out
            if still_active.any():
                r = compute_intrinsic_reward(
                    opt,
                    positions[still_active],
                    zones[still_active],
                    self.intrinsic_scale,
                )
                intrinsic_rewards[still_active] = r

        return intrinsic_rewards

    def _assign_dag_options(self, env_ids: np.ndarray, inventory: np.ndarray):
        """
        Assign the next Option based on the DAG dependency order.

        Falls back to IDLE if no option is available.
        """
        for eid in env_ids:
            inv = inventory[eid:eid+1]
            assigned = False

            # Walk the DAG in priority order
            for opt in [
                Option.COLLECT_WOOD, Option.COLLECT_STONE,
                Option.CRAFT_PICKAXE, Option.MINE_IRON,
                Option.CRAFT_SWORD, Option.CRAFT_ARMOR,
                Option.BUILD_BRIDGE, Option.DEFEAT_ENEMY,
                Option.MINE_GOLD,
            ]:
                if can_initiate(opt, inv)[0] and not is_terminated(opt, inv)[0]:
                    self._set_option(eid, opt)
                    assigned = True
                    break

            if not assigned:
                self._set_option(eid, Option.IDLE)

    def _set_option(self, env_id: int, option: Option):
        """Set a new option for an environment."""
        old_option = self._active_options[env_id]
        self._active_options[env_id] = option
        self._option_step_counts[env_id] = 0
        self._total_option_switches += 1

        self._option_history[env_id].append({
            "from": OPTION_NAMES[old_option],
            "to": OPTION_NAMES[option],
            "steps": int(self._option_step_counts[env_id]),
        })

    def request_new_options_from_llm(
        self, metrics: dict, td_stats: dict, trigger_reason: str
    ):
        """
        Request new Options from the LLM (critic-triggered).

        Instead of adjusting weights, the LLM assigns a high-level option
        for each agent.
        """
        if self.orchestrator is None or self.prompt_builder is None:
            return

        prompt = self.prompt_builder.build_intervention_prompt(
            metrics=metrics,
            td_stats=td_stats,
            trigger_reason=trigger_reason,
            trigger_count=0,
            template="sub_objective",
        )

        response = self.orchestrator.query_intervention(prompt)
        if response is None:
            return

        # Parse LLM response
        a0_option_name = response.get("agent_0_option", "IDLE")
        a1_option_name = response.get("agent_1_option", "IDLE")

        a0_option = Option.from_name(a0_option_name)
        a1_option = Option.from_name(a1_option_name)

        print(f"      [OptionController] LLM assigned: A0={a0_option.name}, A1={a1_option.name}")

        # Apply to all environments
        for eid in range(self.n_envs):
            self._set_option(eid, a0_option)  # Simplified — in practice, per-agent

    def on_episode_reset(self, env_ids: np.ndarray, inventory: np.ndarray):
        """Reset options for environments that terminated."""
        for eid in env_ids:
            self._option_history[eid] = []
        self._assign_dag_options(env_ids, inventory)

    def get_stats(self) -> dict:
        """Return option controller statistics."""
        return {
            "total_switches": self._total_option_switches,
            "option_successes": {
                OPTION_NAMES[i]: int(self._option_success_counts[i])
                for i in range(NUM_OPTIONS)
            },
            "option_failures": {
                OPTION_NAMES[i]: int(self._option_failure_counts[i])
                for i in range(NUM_OPTIONS)
            },
            "current_distribution": {
                OPTION_NAMES[i]: int((self._active_options == i).sum())
                for i in range(NUM_OPTIONS)
            },
        }
