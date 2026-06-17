"""
dataset_generator.py — Extract Successful Trajectories for Fine-Tuning
========================================================================

Scans trajectory data to extract "bottleneck → resolution" pairs from
successful episodes. These become fine-tuning examples for the LLM.

Usage:
    python -m finetune.dataset_generator --input data/trajectories --output data/datasets/raw
"""

import json
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils.replay_buffer import ReplayBuffer


# Subtask dependency order (for bottleneck identification)
SUBTASK_ORDER = [
    "wood", "stone", "pickaxe", "iron", "sword", "armor", "bridge", "enemy", "gold"
]


def identify_bottleneck_states(episode: dict, steps: list) -> list[dict]:
    """
    Identify bottleneck states within a successful episode.

    A bottleneck state is defined as a moment where:
    1. A subtask transition happened (inventory changed)
    2. There was a significant delay before the transition (agent was stuck)

    Returns a list of bottleneck context dicts.
    """
    bottlenecks = []
    timeline = episode.get("subtask_timeline", [])

    if not timeline or not steps:
        return bottlenecks

    prev_step_idx = 0
    for i, event in enumerate(timeline):
        subtask = event["subtask"]
        step_idx = event["step_idx"]

        # Calculate delay since last subtask completion
        delay = step_idx - prev_step_idx

        # If there was a significant delay (>20 steps), this was a bottleneck
        if delay > 20 and step_idx < len(steps):
            # Get the state at the bottleneck (midpoint of the stuck period)
            bottleneck_step_idx = min(
                prev_step_idx + delay // 2,
                len(steps) - 1
            )
            bottleneck_step = steps[bottleneck_step_idx]
            resolution_step = steps[min(step_idx, len(steps) - 1)]

            # Determine which subtasks were incomplete at the bottleneck
            incomplete = []
            for st_name in SUBTASK_ORDER:
                if not bottleneck_step.get("subtasks", {}).get(st_name, False):
                    incomplete.append(st_name)

            # The resolution is the subtask that was eventually completed
            bottlenecks.append({
                "subtask_resolved": subtask,
                "delay_steps": delay,
                "incomplete_subtasks": incomplete,
                "bottleneck_state": {
                    "positions": bottleneck_step.get("positions", []),
                    "inventory": bottleneck_step.get("inventory", []),
                    "subtasks": bottleneck_step.get("subtasks", {}),
                    "step_count": bottleneck_step.get("step_count", 0),
                },
                "resolution_state": {
                    "positions": resolution_step.get("positions", []),
                    "inventory": resolution_step.get("inventory", []),
                    "subtasks": resolution_step.get("subtasks", {}),
                },
            })

        prev_step_idx = step_idx

    return bottlenecks


def generate_training_examples(
    bottlenecks: list[dict],
    episode: dict,
) -> list[dict]:
    """
    Convert bottleneck states into raw training examples.

    Each example contains the bottleneck context and the "expert" resolution
    (what actually happened in the successful trajectory).
    """
    examples = []

    for bn in bottlenecks:
        state = bn["bottleneck_state"]
        incomplete = bn["incomplete_subtasks"]
        resolved = bn["subtask_resolved"]

        # Compute ideal weight adjustment based on what resolved the bottleneck
        # Higher weight for the resolved subtask and its prerequisites
        ideal_weights = _compute_ideal_weights(resolved, incomplete)

        example = {
            "bottleneck_context": {
                "positions": state.get("positions", []),
                "inventory": state.get("inventory", []),
                "subtask_completion": state.get("subtasks", {}),
                "step_count": state.get("step_count", 0),
                "incomplete_subtasks": incomplete,
                "bottleneck_subtask": resolved,
                "delay_steps": bn["delay_steps"],
            },
            "resolution": {
                "subtask_resolved": resolved,
                "ideal_weights": ideal_weights,
                "reasoning": _generate_reasoning(resolved, incomplete),
            },
            "episode_success": episode.get("success", False),
            "episode_length": episode.get("num_steps", 0),
        }
        examples.append(example)

    return examples


def _compute_ideal_weights(resolved_subtask: str, incomplete: list[str]) -> dict:
    """
    Compute ideal reward shaping weights based on what resolved a bottleneck.

    Logic: Give high weight to the resolved subtask and its immediate
    predecessors in the DAG, low weight to already-completed or distant tasks.
    """
    # Map subtask names to weight keys
    subtask_to_weight = {
        "wood": "w_wood",
        "stone": "w_stone",
        "pickaxe": "w_workbench",
        "iron": "w_iron",
        "sword": "w_workbench",
        "armor": "w_workbench",
        "bridge": "w_bridge",
        "enemy": "w_enemy",
        "gold": "w_gold",
    }

    # Prerequisites in the DAG
    prerequisites = {
        "wood": [],
        "stone": [],
        "pickaxe": ["wood", "stone"],
        "iron": ["pickaxe"],
        "sword": ["iron"],
        "armor": ["iron"],
        "bridge": ["wood"],
        "enemy": ["sword", "armor", "bridge"],
        "gold": ["enemy"],
    }

    weights = {
        "w_wood": 0.5, "w_stone": 0.5, "w_workbench": 0.5,
        "w_iron": 0.5, "w_bridge": 0.5, "w_enemy": 0.5, "w_gold": 0.5,
    }

    # Boost the resolved subtask
    target_key = subtask_to_weight.get(resolved_subtask)
    if target_key:
        weights[target_key] = 2.0

    # Boost prerequisites that are still incomplete
    for prereq in prerequisites.get(resolved_subtask, []):
        if prereq in incomplete:
            prereq_key = subtask_to_weight.get(prereq)
            if prereq_key:
                weights[prereq_key] = max(weights[prereq_key], 1.5)

    # Reduce weight for completed subtasks
    for subtask in SUBTASK_ORDER:
        if subtask not in incomplete:
            key = subtask_to_weight.get(subtask)
            if key:
                weights[key] = min(weights[key], 0.3)

    return weights


def _generate_reasoning(resolved: str, incomplete: list[str]) -> str:
    """Generate a reasoning string for the ideal weight adjustment."""
    prereq_str = ", ".join(incomplete[:3]) if incomplete else "none"
    return (
        f"Agents were stuck at {resolved} completion. "
        f"Incomplete prerequisites: {prereq_str}. "
        f"Increasing {resolved} weight to attract agents to the right zone."
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate fine-tuning dataset from trajectories")
    parser.add_argument("--input", type=str, default="data/trajectories")
    parser.add_argument("--output", type=str, default="data/datasets/raw")
    parser.add_argument("--min-subtasks", type=int, default=3,
                        help="Min subtasks completed in successful episodes")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load and index trajectories
    buffer = ReplayBuffer(args.input)
    n_episodes = buffer.build_index()
    if n_episodes == 0:
        print("No trajectory data found. Run training with --enable-logging first.")
        return

    summary = buffer.summary()
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Successful episodes: {summary['successful_episodes']}")
    print(f"Success rate: {summary['success_rate']:.1%}")

    # Get successful episodes
    successes = buffer.get_successful_episodes(min_subtasks=args.min_subtasks)
    print(f"\nProcessing {len(successes)} successful episodes with ≥{args.min_subtasks} subtasks...")

    all_examples = []
    for ep in successes:
        steps = buffer.load_steps_for_episode(ep)
        if not steps:
            continue

        bottlenecks = identify_bottleneck_states(ep, steps)
        examples = generate_training_examples(bottlenecks, ep)
        all_examples.extend(examples)

    # Write raw dataset
    output_path = os.path.join(args.output, "bottleneck_examples.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nGenerated {len(all_examples)} training examples → {output_path}")

    # Statistics
    if all_examples:
        resolved_counts = {}
        for ex in all_examples:
            resolved = ex["bottleneck_context"]["bottleneck_subtask"]
            resolved_counts[resolved] = resolved_counts.get(resolved, 0) + 1
        print("\nBottleneck distribution:")
        for name, count in sorted(resolved_counts.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count} ({count/len(all_examples):.0%})")


if __name__ == "__main__":
    main()
