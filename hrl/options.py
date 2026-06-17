"""
options.py — Option Definitions for Hierarchical RL
=====================================================

Defines the set of high-level Options (macro-actions) and their properties:
    - Termination conditions
    - Intrinsic reward functions
    - Initiation sets (which inventory states allow activation)
    - Agent mapping (which agent executes each option)

Architecture designed for (A) hard-coded options with extensibility for (C)
compound options via the allow_compound_options config flag.
"""

import numpy as np
from enum import IntEnum
from typing import Optional


class Option(IntEnum):
    """Hard-coded Options derived from the crafting DAG."""
    COLLECT_WOOD   = 0
    COLLECT_STONE  = 1
    CRAFT_PICKAXE  = 2
    MINE_IRON      = 3
    CRAFT_SWORD    = 4
    CRAFT_ARMOR    = 5
    BUILD_BRIDGE   = 6
    DEFEAT_ENEMY   = 7
    MINE_GOLD      = 8
    IDLE           = 9

    @classmethod
    def from_name(cls, name: str) -> "Option":
        """Parse an option name string (e.g., 'COLLECT_WOOD') to enum."""
        name = name.upper().replace(" ", "_")
        try:
            return cls[name]
        except KeyError:
            return cls.IDLE


# Number of options (used for one-hot encoding)
NUM_OPTIONS = len(Option)

# Option names for display
OPTION_NAMES = [o.name for o in Option]


# ── Option Properties ────────────────────────────────────────────────────

# Inventory indices (from crafting_env)
I_WOOD, I_STONE, I_IRON, I_PICKAXE = 0, 1, 2, 3
I_SWORD, I_ARMOR, I_GOLD = 4, 5, 6
F_BRIDGE, F_ENEMY_DEFEATED, F_GAME_OVER = 7, 8, 9

# Zone indices mapping
OPTION_TARGET_ZONE = {
    Option.COLLECT_WOOD:  0,   # Wood zone
    Option.COLLECT_STONE: 1,   # Stone zone
    Option.CRAFT_PICKAXE: 2,   # Workbench zone
    Option.MINE_IRON:     3,   # Iron zone
    Option.CRAFT_SWORD:   2,   # Workbench zone
    Option.CRAFT_ARMOR:   2,   # Workbench zone
    Option.BUILD_BRIDGE:  4,   # Bridge zone
    Option.DEFEAT_ENEMY:  5,   # Enemy zone
    Option.MINE_GOLD:     6,   # Gold zone
    Option.IDLE:          -1,  # No target
}

# Which agent should execute each option
# 0 = Agent 0 (Lumberjack), 1 = Agent 1 (Miner), 2 = Both
OPTION_AGENT = {
    Option.COLLECT_WOOD:  0,
    Option.COLLECT_STONE: 1,
    Option.CRAFT_PICKAXE: 2,   # Both need to be at workbench
    Option.MINE_IRON:     1,
    Option.CRAFT_SWORD:   2,   # Both at workbench
    Option.CRAFT_ARMOR:   2,   # Both at workbench
    Option.BUILD_BRIDGE:  2,   # Both needed
    Option.DEFEAT_ENEMY:  2,   # Both needed
    Option.MINE_GOLD:     1,
    Option.IDLE:          2,
}


def can_initiate(option: Option, inventory: np.ndarray) -> np.ndarray:
    """
    Check which environments can initiate a given Option.

    Args:
        option: The Option to check.
        inventory: [n_envs, NUM_ITEMS] inventory state.

    Returns:
        [n_envs] bool array — True where the option can be started.
    """
    n = inventory.shape[0]

    if option == Option.COLLECT_WOOD:
        # Can always collect wood if not all mined (handled by env limits)
        return np.ones(n, dtype=bool)

    elif option == Option.COLLECT_STONE:
        return np.ones(n, dtype=bool)

    elif option == Option.CRAFT_PICKAXE:
        # Need Wood ≥ 1 and Stone ≥ 1, Pickaxe not yet crafted
        return (
            (inventory[:, I_WOOD] >= 1) &
            (inventory[:, I_STONE] >= 1) &
            (inventory[:, I_PICKAXE] < 1)
        )

    elif option == Option.MINE_IRON:
        return inventory[:, I_PICKAXE] >= 1

    elif option == Option.CRAFT_SWORD:
        return (
            (inventory[:, I_IRON] >= 1) &
            (inventory[:, I_SWORD] < 1)
        )

    elif option == Option.CRAFT_ARMOR:
        return (
            (inventory[:, I_IRON] >= 1) &
            (inventory[:, I_ARMOR] < 1)
        )

    elif option == Option.BUILD_BRIDGE:
        return (
            (inventory[:, I_WOOD] >= 1) &
            (inventory[:, F_BRIDGE] < 1)
        )

    elif option == Option.DEFEAT_ENEMY:
        return (
            (inventory[:, F_BRIDGE] >= 1) &
            (inventory[:, I_SWORD] >= 1) &
            (inventory[:, I_ARMOR] >= 1) &
            (inventory[:, F_ENEMY_DEFEATED] < 1)
        )

    elif option == Option.MINE_GOLD:
        return (
            (inventory[:, F_ENEMY_DEFEATED] >= 1) &
            (inventory[:, I_GOLD] < 1)
        )

    elif option == Option.IDLE:
        return np.ones(n, dtype=bool)

    return np.zeros(n, dtype=bool)


def is_terminated(option: Option, inventory: np.ndarray) -> np.ndarray:
    """
    Check if an Option has terminated (completed its objective).

    Args:
        option: The active Option.
        inventory: [n_envs, NUM_ITEMS] current inventory.

    Returns:
        [n_envs] bool array — True where the option is complete.
    """
    n = inventory.shape[0]

    if option == Option.COLLECT_WOOD:
        return inventory[:, I_WOOD] >= 1

    elif option == Option.COLLECT_STONE:
        return inventory[:, I_STONE] >= 1

    elif option == Option.CRAFT_PICKAXE:
        return inventory[:, I_PICKAXE] >= 1

    elif option == Option.MINE_IRON:
        return inventory[:, I_IRON] >= 1

    elif option == Option.CRAFT_SWORD:
        return inventory[:, I_SWORD] >= 1

    elif option == Option.CRAFT_ARMOR:
        return inventory[:, I_ARMOR] >= 1

    elif option == Option.BUILD_BRIDGE:
        return inventory[:, F_BRIDGE] >= 1

    elif option == Option.DEFEAT_ENEMY:
        return inventory[:, F_ENEMY_DEFEATED] >= 1

    elif option == Option.MINE_GOLD:
        return inventory[:, I_GOLD] >= 1

    elif option == Option.IDLE:
        return np.zeros(n, dtype=bool)  # Never self-terminates

    return np.zeros(n, dtype=bool)


def compute_intrinsic_reward(
    option: Option,
    agent_positions: np.ndarray,
    env_zones: np.ndarray,
    intrinsic_scale: float = 0.1,
) -> np.ndarray:
    """
    Compute dense intrinsic reward for the active option.

    Distance-based: reward for getting closer to the target zone.

    Args:
        option: Active option.
        agent_positions: [n_envs, 2, 2] agent positions.
        env_zones: [n_envs, 7, 2] zone positions.
        intrinsic_scale: Scale factor for intrinsic reward.

    Returns:
        [n_envs, 2] intrinsic rewards per agent.
    """
    n = agent_positions.shape[0]
    target_zone = OPTION_TARGET_ZONE.get(option, -1)
    target_agent = OPTION_AGENT.get(option, 2)

    rewards = np.zeros((n, 2), dtype=np.float32)

    if target_zone < 0:
        return rewards

    target_pos = env_zones[:, target_zone]  # [n_envs, 2]

    for a in range(2):
        if target_agent == 2 or target_agent == a:
            dist = np.linalg.norm(agent_positions[:, a] - target_pos, axis=1)
            # Negative distance as reward (closer = better)
            rewards[:, a] = -dist * intrinsic_scale

    return rewards


def option_to_one_hot(options: np.ndarray) -> np.ndarray:
    """
    Convert option indices to one-hot vectors.

    Args:
        options: [n] int array of option indices.

    Returns:
        [n, NUM_OPTIONS] one-hot encoded options.
    """
    n = options.shape[0]
    one_hot = np.zeros((n, NUM_OPTIONS), dtype=np.float32)
    one_hot[np.arange(n), options] = 1.0
    return one_hot


# ── Compound Option Support (Phase 4C) ───────────────────────────────────

class CompoundOption:
    """
    A compound option composed of multiple sequential primitive options.

    Used when allow_compound_options=True (Phase 4C evaluation).
    """

    def __init__(self, name: str, sequence: list[Option]):
        self.name = name
        self.sequence = sequence
        self._current_idx = 0

    @property
    def current_option(self) -> Option:
        if self._current_idx < len(self.sequence):
            return self.sequence[self._current_idx]
        return Option.IDLE

    def advance(self) -> bool:
        """Advance to the next option in the sequence. Returns True if complete."""
        self._current_idx += 1
        return self._current_idx >= len(self.sequence)

    def reset(self):
        self._current_idx = 0

    @property
    def is_complete(self) -> bool:
        return self._current_idx >= len(self.sequence)


# Pre-defined compound options (for Phase 4C)
COMPOUND_OPTIONS = {
    "RUSH_BRIDGE": CompoundOption("RUSH_BRIDGE", [
        Option.COLLECT_WOOD, Option.BUILD_BRIDGE
    ]),
    "FULL_EQUIP": CompoundOption("FULL_EQUIP", [
        Option.MINE_IRON, Option.CRAFT_SWORD, Option.CRAFT_ARMOR
    ]),
    "EARLY_GAME": CompoundOption("EARLY_GAME", [
        Option.COLLECT_WOOD, Option.COLLECT_STONE, Option.CRAFT_PICKAXE
    ]),
}
