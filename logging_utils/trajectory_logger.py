"""
trajectory_logger.py — Per-Step State Capture for Trajectory Collection
========================================================================

Captures structured episode data at each step for a subset of environments.
Data is written as JSONL with gzip compression for efficient storage.

Usage:
    logger = TrajectoryLogger("data/trajectories")
    # Inside training loop:
    logger.log_step(update, step, env_ids, snapshot, actions, ...)
    logger.log_episode_end(env_ids, terminal_flags, success)
    # End:
    logger.close()
"""

import json
import gzip
import os
import time
import numpy as np
from typing import Optional


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


class TrajectoryLogger:
    """
    Logs per-step environment state for trajectory collection.

    Maintains per-environment episode buffers. When an episode ends,
    the complete trajectory is flushed to disk.

    Args:
        output_dir: Directory to write trajectory files.
        buffer_size: Max episodes to hold in memory before flushing.
    """

    def __init__(self, output_dir: str = "data/trajectories", buffer_size: int = 1000):
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._step_file = gzip.open(
            os.path.join(output_dir, f"steps_{timestamp}.jsonl.gz"), "wt", encoding="utf-8"
        )
        self._episode_file = gzip.open(
            os.path.join(output_dir, f"episodes_{timestamp}.jsonl.gz"), "wt", encoding="utf-8"
        )

        # Per-env episode buffers: env_id -> list of step dicts
        self._episode_buffers: dict[int, list] = {}
        self._episode_counts: dict[int, int] = {}
        self._total_episodes = 0

        print(f"[TrajectoryLogger] Writing to {output_dir}/")

    def log_step(
        self,
        update: int,
        step: int,
        env_ids: np.ndarray,
        snapshot: dict,
        actions: np.ndarray,
        env_rewards: np.ndarray,
        shaped_rewards: np.ndarray,
        goal_zones: np.ndarray,
        goal_active: np.ndarray,
        terminal: np.ndarray,
        td_errors: Optional[np.ndarray] = None,
    ):
        """
        Log a single step for multiple environments.

        Args:
            update: Current training update number.
            step: Step within the current rollout.
            env_ids: [K] environment indices being logged.
            snapshot: dict from env.get_state_snapshot(env_ids).
            actions: [K, 2] actions taken.
            env_rewards: [K, 2] environment rewards.
            shaped_rewards: [K, 2] shaped rewards.
            goal_zones: [K, 2] goal zone indices.
            goal_active: [K, 2] goal active flags.
            terminal: [K] terminal flags.
            td_errors: Optional [K, 2] TD errors at this step.
        """
        for i, env_id in enumerate(env_ids):
            eid = int(env_id)

            step_data = {
                "update": update,
                "step": step,
                "env_id": eid,
                "positions": snapshot["positions"][i].tolist(),
                "inventory": snapshot["inventory"][i].tolist(),
                "step_count": int(snapshot["step_counts"][i]),
                "actions": actions[i].tolist(),
                "env_reward": env_rewards[i].tolist(),
                "shaped_reward": shaped_rewards[i].tolist(),
                "goal_zones": goal_zones[i].tolist(),
                "goal_active": goal_active[i].tolist(),
            }

            if td_errors is not None:
                step_data["td_errors"] = td_errors[i].tolist()

            # Subtask progress (compact)
            sp = snapshot["subtask_progress"]
            step_data["subtasks"] = {
                k: bool(v[i]) for k, v in sp.items()
            }

            # Append to per-env episode buffer
            if eid not in self._episode_buffers:
                self._episode_buffers[eid] = []
            self._episode_buffers[eid].append(step_data)

            # Write to step-level file
            self._step_file.write(json.dumps(step_data, cls=NumpyEncoder) + "\n")

    def log_episode_end(
        self,
        env_ids: np.ndarray,
        terminal_flags: np.ndarray,
        success: np.ndarray,
    ):
        """
        Called when episodes terminate. Flushes complete episode data.

        Args:
            env_ids: [K] environment indices that terminated.
            terminal_flags: [K, NUM_ITEMS] terminal inventory flags.
            success: [K] bool — whether gold was mined.
        """
        for i, env_id in enumerate(env_ids):
            eid = int(env_id)
            episode_steps = self._episode_buffers.pop(eid, [])

            if not episode_steps:
                continue

            episode_record = {
                "env_id": eid,
                "episode_num": self._episode_counts.get(eid, 0),
                "num_steps": len(episode_steps),
                "success": bool(success[i]),
                "terminal_flags": terminal_flags[i].tolist(),
                "first_update": episode_steps[0]["update"],
                "last_update": episode_steps[-1]["update"],
                # Summarize the trajectory as the sequence of subtask completions
                "subtask_timeline": self._extract_subtask_timeline(episode_steps),
            }

            self._episode_file.write(json.dumps(episode_record, cls=NumpyEncoder) + "\n")

            self._episode_counts[eid] = self._episode_counts.get(eid, 0) + 1
            self._total_episodes += 1

            # Periodic flush
            if self._total_episodes % 100 == 0:
                self._step_file.flush()
                self._episode_file.flush()

    def _extract_subtask_timeline(self, steps: list) -> list:
        """
        Extract the order in which subtasks were completed during an episode.

        Returns a list of {"subtask": name, "step_idx": int} dicts.
        """
        timeline = []
        prev_subtasks = {}
        for idx, step in enumerate(steps):
            for name, completed in step.get("subtasks", {}).items():
                if completed and not prev_subtasks.get(name, False):
                    timeline.append({"subtask": name, "step_idx": idx})
            prev_subtasks = step.get("subtasks", {})
        return timeline

    def close(self):
        """Flush and close all files."""
        self._step_file.flush()
        self._episode_file.flush()
        self._step_file.close()
        self._episode_file.close()
        print(f"[TrajectoryLogger] Closed. Total episodes logged: {self._total_episodes}")
