"""
replay_buffer.py — Successful Trajectory Index & Retrieval
===========================================================

Scans trajectory data on disk and provides filtered retrieval:
    - Success filter (gold_mined episodes)
    - Milestone filter (reached specific subtask)
    - Lazy loading: only indexes episode metadata, loads full trajectories on demand

Usage:
    buffer = ReplayBuffer("data/trajectories")
    buffer.build_index()
    episodes = buffer.get_successful_episodes(min_subtasks=5)
"""

import json
import gzip
import glob
import os
from typing import Optional


class ReplayBuffer:
    """
    Indexes and retrieves episode trajectories from logged data.

    The buffer maintains an in-memory index of episode metadata (from episode files)
    and lazily loads full step data from step files when requested.
    """

    def __init__(self, data_dir: str = "data/trajectories"):
        self.data_dir = data_dir
        self._episode_index: list[dict] = []
        self._step_files: list[str] = []

    def build_index(self) -> int:
        """
        Scan all episode files and build an in-memory index.

        Returns:
            Number of episodes indexed.
        """
        self._episode_index = []
        episode_files = sorted(glob.glob(os.path.join(self.data_dir, "episodes_*.jsonl.gz")))
        self._step_files = sorted(glob.glob(os.path.join(self.data_dir, "steps_*.jsonl.gz")))

        for ep_file in episode_files:
            with gzip.open(ep_file, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["_source_file"] = ep_file
                    self._episode_index.append(record)

        print(f"[ReplayBuffer] Indexed {len(self._episode_index)} episodes "
              f"from {len(episode_files)} files")
        return len(self._episode_index)

    def get_successful_episodes(self, min_subtasks: int = 0) -> list[dict]:
        """
        Return all episodes where gold was mined (success=True).

        Args:
            min_subtasks: Minimum number of subtasks completed in the episode
                          (useful for filtering partially successful runs).

        Returns:
            List of episode metadata dicts.
        """
        results = []
        for ep in self._episode_index:
            if not ep.get("success", False):
                continue
            timeline = ep.get("subtask_timeline", [])
            if len(timeline) >= min_subtasks:
                results.append(ep)
        return results

    def get_episodes_by_milestone(self, milestone: str) -> list[dict]:
        """
        Return episodes that reached a specific subtask milestone.

        Args:
            milestone: Subtask name (e.g., "pickaxe", "bridge", "gold").

        Returns:
            List of episode metadata dicts.
        """
        results = []
        for ep in self._episode_index:
            timeline = ep.get("subtask_timeline", [])
            subtask_names = [t["subtask"] for t in timeline]
            if milestone in subtask_names:
                results.append(ep)
        return results

    def get_failed_episodes(self, max_subtasks: Optional[int] = None) -> list[dict]:
        """
        Return episodes where gold was NOT mined.

        Args:
            max_subtasks: If set, only return failures that completed at most
                          this many subtasks (i.e., episodes that got stuck early).
        """
        results = []
        for ep in self._episode_index:
            if ep.get("success", False):
                continue
            if max_subtasks is not None:
                timeline = ep.get("subtask_timeline", [])
                if len(timeline) > max_subtasks:
                    continue
            results.append(ep)
        return results

    def load_steps_for_episode(self, episode: dict) -> list[dict]:
        """
        Load all step-level data for a specific episode.

        This performs a linear scan of step files filtered by env_id and update range.
        For large datasets, consider building a step-level index.

        Args:
            episode: Episode metadata dict from the index.

        Returns:
            List of step dicts in chronological order.
        """
        env_id = episode["env_id"]
        first_update = episode["first_update"]
        last_update = episode["last_update"]

        steps = []
        for step_file in self._step_files:
            with gzip.open(step_file, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if (record["env_id"] == env_id and
                        first_update <= record["update"] <= last_update):
                        steps.append(record)

        # Sort by (update, step)
        steps.sort(key=lambda s: (s["update"], s["step"]))
        return steps

    def summary(self) -> dict:
        """Return summary statistics about the indexed episodes."""
        total = len(self._episode_index)
        successful = sum(1 for ep in self._episode_index if ep.get("success", False))
        avg_steps = (
            sum(ep.get("num_steps", 0) for ep in self._episode_index) / max(total, 1)
        )

        # Milestone distribution
        milestone_counts = {}
        for ep in self._episode_index:
            for t in ep.get("subtask_timeline", []):
                name = t["subtask"]
                milestone_counts[name] = milestone_counts.get(name, 0) + 1

        return {
            "total_episodes": total,
            "successful_episodes": successful,
            "success_rate": successful / max(total, 1),
            "avg_steps_per_episode": avg_steps,
            "milestone_counts": milestone_counts,
        }


if __name__ == "__main__":
    buf = ReplayBuffer("data/trajectories")
    n = buf.build_index()
    if n > 0:
        print(buf.summary())
        successes = buf.get_successful_episodes(min_subtasks=5)
        print(f"Successful episodes with ≥5 subtasks: {len(successes)}")
    else:
        print("No trajectory data found. Run training with --enable-logging first.")
