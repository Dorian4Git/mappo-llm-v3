"""
crafting_env.py — Extended Crafting Environment (Inherited from V2)
===================================================================

Dependency DAG (Inventory-based):
    [Wood (A0)] & [Stone (A1)]
         ↓            ↓
       [Workbench → Pickaxe (either)]
         ↓            ↓
    [Bridge (A0)]  [Iron (A1, needs Pickaxe)]
         ↓            ↓
         ↓     [Sword & Armor (either, needs Iron)]
         ↓            ↓
       [Gold (A1, needs Bridge AND Sword AND Armor)]

Observation (14-dim local):
    [a0.x, a0.y, a1.x, a1.y, wood, stone, iron, pickaxe, sword, armor, gold, bridge, enemy_def, game_over]

V3 Additions:
    - get_state_snapshot() method for trajectory logging
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_LIMIT     = 60.0
MAX_STEPS      = 800
DIST_THRESHOLD = 3.0
STEP_PENALTY   = -0.01

ZONES = {
    "wood":      np.array([10.0, 10.0], dtype=np.float32),
    "stone":     np.array([25.0, 10.0], dtype=np.float32),
    "workbench": np.array([20.0, 30.0], dtype=np.float32),
    "iron":      np.array([15.0, 45.0], dtype=np.float32),
    "bridge":    np.array([35.0, 40.0], dtype=np.float32),
    "enemy":     np.array([45.0, 40.0], dtype=np.float32),
    "gold":      np.array([50.0, 20.0], dtype=np.float32),
}

OBSTACLES: list[tuple[float, float, float, float]] = [
    (20.0, 20.0, 40.0, 22.0),
    (20.0, 35.0, 22.0, 50.0),
    (40.0, 35.0, 42.0, 50.0),
]

RIVER_X_MIN, RIVER_X_MAX = 34.0, 36.0
BRIDGE_Y_MIN, BRIDGE_Y_MAX = 38.0, 42.0

# Inventory indices
I_WOOD = 0
I_STONE = 1
I_IRON = 2
I_PICKAXE = 3
I_SWORD = 4
I_ARMOR = 5
I_GOLD = 6
F_BRIDGE = 7
F_ENEMY_DEFEATED = 8
F_GAME_OVER = 9
NUM_ITEMS = 10

# Subtask name mapping (for logging and display)
ITEM_NAMES = [
    "Wood", "Stone", "Iron", "Pickaxe", "Sword",
    "Armor", "Gold", "Bridge", "Enemy", "GameOver",
]


# ---------------------------------------------------------------------------
# BatchCraftingEnvV2 — Fully Vectorized
# ---------------------------------------------------------------------------
class BatchCraftingEnvV2:
    def __init__(self, n_envs: int = 32, seed: Optional[int] = None):
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)

        self.pos = np.zeros((n_envs, 2, 2), dtype=np.float32)
        self.inventory = np.zeros((n_envs, NUM_ITEMS), dtype=np.int32)
        self.step_counts = np.zeros(n_envs, dtype=np.int32)
        
        # Track total gathered to prevent infinite reward farming
        self.total_mined = np.zeros((n_envs, 3), dtype=np.int32) # wood, stone, iron

        self.zones = np.zeros((n_envs, 7, 2), dtype=np.float32)
        self._move_deltas = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]], dtype=np.float32)
        self._obstacles = np.array(OBSTACLES, dtype=np.float32)

        self._zone_bases = np.array([
            ZONES["wood"], ZONES["stone"], ZONES["workbench"],
            ZONES["iron"], ZONES["bridge"], ZONES["enemy"], ZONES["gold"]
        ])

        self.reset()

    def _spawn_positions(self, n: int) -> np.ndarray:
        pos = np.zeros((n, 2, 2), dtype=np.float32)
        pos[:, 0, 0] = self.rng.integers(2, 19, size=n).astype(np.float32)
        pos[:, 0, 1] = self.rng.integers(2, 59, size=n).astype(np.float32)
        pos[:, 1, 0] = self.rng.integers(2, 19, size=n).astype(np.float32)
        pos[:, 1, 1] = self.rng.integers(2, 59, size=n).astype(np.float32)
        return pos

    def reset(self) -> tuple[np.ndarray, list]:
        self.pos = self._spawn_positions(self.n_envs)
        self.inventory[:] = 0
        self.total_mined[:] = 0
        self.step_counts[:] = 0
        for i in range(7):
            self.zones[:, i, :] = self._zone_bases[i]
        return self._get_obs_batch(), [{} for _ in range(self.n_envs)]

    def _get_obs_batch(self) -> np.ndarray:
        row = np.concatenate([
            self.pos[:, 0],
            self.pos[:, 1],
            self.inventory.astype(np.float32),
        ], axis=1)
        return np.stack([row, row], axis=1)

    def get_state_snapshot(self, env_ids: np.ndarray = None) -> dict:
        """
        Return a structured snapshot of the environment state for trajectory logging.

        Args:
            env_ids: Optional array of environment indices to snapshot.
                     If None, snapshots all environments.

        Returns:
            dict with keys:
                - positions: [n, 2, 2] agent positions
                - inventory: [n, NUM_ITEMS] inventory state
                - step_counts: [n] step counts
                - total_mined: [n, 3] cumulative resource mining counts
                - zones: [n, 7, 2] zone positions (enemy may have moved)
                - subtask_progress: dict mapping subtask name -> bool array [n]
        """
        if env_ids is None:
            env_ids = np.arange(self.n_envs)

        inv = self.inventory[env_ids]
        subtask_progress = {
            "wood":     (inv[:, I_WOOD] > 0) | (inv[:, I_PICKAXE] > 0) | (inv[:, F_BRIDGE] > 0),
            "stone":    (inv[:, I_STONE] > 0) | (inv[:, I_PICKAXE] > 0),
            "pickaxe":  inv[:, I_PICKAXE] > 0,
            "iron":     (inv[:, I_IRON] > 0) | (inv[:, I_SWORD] > 0) | (inv[:, I_ARMOR] > 0),
            "sword":    inv[:, I_SWORD] > 0,
            "armor":    inv[:, I_ARMOR] > 0,
            "bridge":   inv[:, F_BRIDGE] > 0,
            "enemy":    inv[:, F_ENEMY_DEFEATED] > 0,
            "gold":     inv[:, I_GOLD] > 0,
        }

        return {
            "positions":         self.pos[env_ids].copy(),
            "inventory":         inv.copy(),
            "step_counts":       self.step_counts[env_ids].copy(),
            "total_mined":       self.total_mined[env_ids].copy(),
            "zones":             self.zones[env_ids].copy(),
            "subtask_progress":  subtask_progress,
        }

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        self.step_counts += 1
        rewards = np.full((self.n_envs, 2), STEP_PENALTY, dtype=np.float32)

        # ── Movement ────────────────────────────────────────
        deltas = self._move_deltas[actions]
        new_pos = self.pos + deltas

        in_bounds = (
            (new_pos[:, :, 0] >= 0) & (new_pos[:, :, 0] <= GRID_LIMIT) &
            (new_pos[:, :, 1] >= 0) & (new_pos[:, :, 1] <= GRID_LIMIT)
        )

        x = new_pos[:, :, 0:1]
        y = new_pos[:, :, 1:2]
        
        blocked = np.zeros((self.n_envs, 2, 1), dtype=bool)
        if len(self._obstacles) > 0:
            ox0, oy0, ox1, oy1 = self._obstacles[:, 0], self._obstacles[:, 1], self._obstacles[:, 2], self._obstacles[:, 3]
            in_obs = (x >= ox0) & (x <= ox1) & (y >= oy0) & (y <= oy1)
            blocked |= in_obs.any(axis=2, keepdims=True)
            
        in_river_x = (x >= RIVER_X_MIN) & (x <= RIVER_X_MAX)
        in_bridge_y = (y >= BRIDGE_Y_MIN) & (y <= BRIDGE_Y_MAX)
        bridge_active = self.inventory[:, F_BRIDGE, np.newaxis, np.newaxis] > 0
        
        blocked_by_river = in_river_x & ~(in_bridge_y & bridge_active)
        blocked |= blocked_by_river
        blocked = blocked.squeeze(-1)

        can_move = (actions < 4) & in_bounds & ~blocked
        self.pos = np.where(can_move[:, :, np.newaxis], new_pos, self.pos)

        # ── Interactions ─────────────────────────────────────────────────
        interact_a0 = (actions[:, 0] == 4)
        interact_a1 = (actions[:, 1] == 4)

        def dist_a(agent_idx, zone_idx):
            return np.linalg.norm(self.pos[:, agent_idx] - self.zones[:, zone_idx], axis=1)

        def both_at(zone):
            return (dist_a(0, zone) < DIST_THRESHOLD) & (dist_a(1, zone) < DIST_THRESHOLD)

        # Wood (A0 only, max 2 total mined)
        wood_ok = interact_a0 & (self.total_mined[:, 0] < 2) & (dist_a(0, 0) < DIST_THRESHOLD)
        self.inventory[wood_ok, I_WOOD] += 1
        self.total_mined[wood_ok, 0] += 1
        rewards[wood_ok, :] += 2.0

        # Stone (A1 only, max 1 total mined)
        stone_ok = interact_a1 & (self.total_mined[:, 1] < 1) & (dist_a(1, 1) < DIST_THRESHOLD)
        self.inventory[stone_ok, I_STONE] += 1
        self.total_mined[stone_ok, 1] += 1
        rewards[stone_ok, :] += 2.0

        # Pickaxe (both at workbench, costs 1W, 1S)
        wb_ok_a0 = interact_a0 & (self.inventory[:, I_WOOD] >= 1) & (self.inventory[:, I_STONE] >= 1) & (self.inventory[:, I_PICKAXE] < 1) & both_at(2)
        wb_ok_a1 = interact_a1 & (self.inventory[:, I_WOOD] >= 1) & (self.inventory[:, I_STONE] >= 1) & (self.inventory[:, I_PICKAXE] < 1) & both_at(2)
        wb_ok = wb_ok_a0 | wb_ok_a1
        self.inventory[wb_ok, I_WOOD] -= 1
        self.inventory[wb_ok, I_STONE] -= 1
        self.inventory[wb_ok, I_PICKAXE] += 1
        rewards[wb_ok, :] += 3.0

        # Iron (A1 only, needs pickaxe, max 2 total mined)
        iron_ok = interact_a1 & (self.inventory[:, I_PICKAXE] >= 1) & (self.total_mined[:, 2] < 2) & (dist_a(1, 3) < DIST_THRESHOLD)
        self.inventory[iron_ok, I_IRON] += 1
        self.total_mined[iron_ok, 2] += 1
        rewards[iron_ok, :] += 2.0

        # Sword (both at workbench, costs 1 Iron)
        sword_ok = interact_a1 & (self.inventory[:, I_IRON] >= 1) & (self.inventory[:, I_SWORD] < 1) & (dist_a(1, 2) < DIST_THRESHOLD) & both_at(2)
        self.inventory[sword_ok, I_IRON] -= 1
        self.inventory[sword_ok, I_SWORD] += 1
        rewards[sword_ok, :] += 3.0

        # Armor (both at workbench, costs 1 Iron)
        armor_ok = interact_a1 & (self.inventory[:, I_IRON] >= 1) & (self.inventory[:, I_ARMOR] < 1) & (dist_a(1, 2) < DIST_THRESHOLD) & both_at(2)
        self.inventory[armor_ok, I_IRON] -= 1
        self.inventory[armor_ok, I_ARMOR] += 1
        rewards[armor_ok, :] += 3.0

        # Bridge (A0 triggers, requires both at bridge, costs 1W)
        bridge_ok = interact_a0 & (self.inventory[:, I_WOOD] >= 1) & (self.inventory[:, F_BRIDGE] == 0) & (dist_a(0, 4) < DIST_THRESHOLD) & both_at(4)
        self.inventory[bridge_ok, I_WOOD] -= 1
        self.inventory[bridge_ok, F_BRIDGE] = 1
        rewards[bridge_ok, :] += 3.0

        # Enemy
        enemy_interact = interact_a1 & (self.inventory[:, F_BRIDGE] == 1) & (self.inventory[:, F_ENEMY_DEFEATED] == 0) & (self.inventory[:, F_GAME_OVER] == 0) & (dist_a(1, 5) < DIST_THRESHOLD)
        enemy_success = enemy_interact & (self.inventory[:, I_SWORD] >= 1) & (self.inventory[:, I_ARMOR] >= 1) & both_at(5)
        self.inventory[enemy_success, F_ENEMY_DEFEATED] = 1
        rewards[enemy_success, :] += 10.0

        enemy_fail = enemy_interact & ~enemy_success
        rewards[enemy_fail, :] -= 2.0

        # Gold
        gold_ok = interact_a1 & (self.inventory[:, F_ENEMY_DEFEATED] == 1) & (self.inventory[:, I_GOLD] == 0) & (dist_a(1, 6) < DIST_THRESHOLD)
        self.inventory[gold_ok, I_GOLD] = 1
        rewards[gold_ok, :] += 15.0

        # ── Terminal conditions ──────────────────────────────────────────
        dones  = (self.inventory[:, I_GOLD] == 1) | (self.inventory[:, F_GAME_OVER] == 1)
        truncs = (self.step_counts >= MAX_STEPS)
        terminal = dones | truncs

        terminal_flags = self.inventory.copy()
        # Ensure consumed items still register as completed for subtask metrics
        terminal_flags[:, I_STONE] += terminal_flags[:, I_PICKAXE]
        terminal_flags[:, I_WOOD] += terminal_flags[:, I_PICKAXE] + terminal_flags[:, F_BRIDGE]
        terminal_flags[:, I_IRON] += terminal_flags[:, I_SWORD] + terminal_flags[:, I_ARMOR]

        if terminal.any():
            n_reset = int(terminal.sum())
            self.pos[terminal] = self._spawn_positions(n_reset)
            self.inventory[terminal] = 0
            self.total_mined[terminal] = 0
            self.step_counts[terminal] = 0

        # Enemy Random Walk (Stationary until detected, then slow)
        dist_a0_enemy = np.linalg.norm(self.pos[:, 0] - self.zones[:, 5], axis=1)
        dist_a1_enemy = np.linalg.norm(self.pos[:, 1] - self.zones[:, 5], axis=1)
        detected = (dist_a0_enemy < 7.0) | (dist_a1_enemy < 7.0)
        
        move_mask = self.rng.random(self.n_envs) < 0.05  # Slow (5% vs 20%)
        move_mask &= (self.inventory[:, F_ENEMY_DEFEATED] == 0)
        move_mask &= detected
        
        if move_mask.any():
            n_move = move_mask.sum()
            dx = self.rng.integers(-1, 2, size=n_move).astype(np.float32)
            dy = self.rng.integers(-1, 2, size=n_move).astype(np.float32)
            
            new_ex = self.zones[move_mask, 5, 0] + dx
            new_ey = self.zones[move_mask, 5, 1] + dy
            
            new_ex = np.clip(new_ex, RIVER_X_MAX + 1, GRID_LIMIT - 1)
            new_ey = np.clip(new_ey, 0, GRID_LIMIT - 1)
            
            self.zones[move_mask, 5, 0] = new_ex
            self.zones[move_mask, 5, 1] = new_ey

        obs = self._get_obs_batch()
        return obs, rewards, dones, truncs, {'terminal_flags': terminal_flags}

    def _get_obs_batch_fov(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            fov_crops: [n_envs, 2, 9, 7, 7] - Local agent observations
            global_map: [n_envs, 9, 60, 60] - Global state for CTDE
        """
        n = self.n_envs
        global_map = np.zeros((n, 9, 61, 61), dtype=np.float32)
        
        px0 = np.clip(self.pos[:, 0, 0].astype(int), 0, 60)
        py0 = np.clip(self.pos[:, 0, 1].astype(int), 0, 60)
        px1 = np.clip(self.pos[:, 1, 0].astype(int), 0, 60)
        py1 = np.clip(self.pos[:, 1, 1].astype(int), 0, 60)
        env_indices = np.arange(n)
        global_map[env_indices, 1, px0, py0] = 1.0
        global_map[env_indices, 1, px1, py1] = 1.0
        
        for (x0, y0, x1, y1) in self._obstacles:
            global_map[:, 0, int(x0):int(x1)+1, int(y0):int(y1)+1] = 1.0
            
        global_map[:, 0, int(RIVER_X_MIN):int(RIVER_X_MAX)+1, :] = 1.0
        bridge_active = self.inventory[:, F_BRIDGE] > 0
        global_map[bridge_active, 0, int(RIVER_X_MIN):int(RIVER_X_MAX)+1, int(BRIDGE_Y_MIN):int(BRIDGE_Y_MAX)+1] = 0.0
        
        # We place zones on the map if they haven't been completely consumed
        # Zones: 0:Wood, 1:Stone, 2:WB, 3:Iron, 4:Bridge, 5:Enemy, 6:Gold
        zone_to_channel = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 2, 6: 8}
        
        gathered_mask = np.zeros((n, 7), dtype=bool)
        gathered_mask[:, 0] = self.total_mined[:, 0] >= 2 # Wood
        gathered_mask[:, 1] = self.total_mined[:, 1] >= 1 # Stone
        gathered_mask[:, 3] = self.total_mined[:, 2] >= 2 # Iron
        gathered_mask[:, 5] = self.inventory[:, F_ENEMY_DEFEATED] > 0
        gathered_mask[:, 6] = self.inventory[:, I_GOLD] > 0
        
        for z_idx, c_idx in zone_to_channel.items():
            valid_envs = ~gathered_mask[:, z_idx]
            if valid_envs.any():
                zx = self.zones[valid_envs, z_idx, 0].astype(int)
                zy = self.zones[valid_envs, z_idx, 1].astype(int)
                for dx_z in [-1, 0, 1]:
                    for dy_z in [-1, 0, 1]:
                        nx = np.clip(zx + dx_z, 0, 60)
                        ny = np.clip(zy + dy_z, 0, 60)
                        global_map[valid_envs, c_idx, nx, ny] = 1.0
        
        padded_map = np.pad(global_map, ((0,0), (0,0), (3,3), (3,3)), mode='constant', constant_values=0)
        padded_map[:, 0, :3, :] = 1.0
        padded_map[:, 0, -3:, :] = 1.0
        padded_map[:, 0, :, :3] = 1.0
        padded_map[:, 0, :, -3:] = 1.0
        
        crops = np.zeros((n, 2, 9, 7, 7), dtype=np.float32)
        env_grid = np.arange(n)[:, None, None, None]
        ch_grid = np.arange(9)[None, :, None, None]
        dx = np.arange(7)
        dy = np.arange(7)
        
        for a in range(2):
            px = self.pos[:, a, 0].astype(int)
            py = self.pos[:, a, 1].astype(int)
            
            X_full = (px[:, None, None] + dx[None, :, None])[:, None, :, :]
            Y_full = (py[:, None, None] + dy[None, None, :])[:, None, :, :]
            
            crops[:, a] = padded_map[env_grid, ch_grid, X_full, Y_full]
            
        return crops, global_map

    def close(self):
        pass


if __name__ == "__main__":
    env = BatchCraftingEnvV2(n_envs=2, seed=42)
    obs, _ = env.reset()
    fov, gmap = env._get_obs_batch_fov()
    print("FOV:", fov.shape)
    print("Global map:", gmap.shape)
    print("Obs shape:", obs.shape)

    snapshot = env.get_state_snapshot()
    print("Snapshot keys:", list(snapshot.keys()))
    print("Subtask progress keys:", list(snapshot["subtask_progress"].keys()))
    print("Inventory OK.")
