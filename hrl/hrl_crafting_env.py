import numpy as np
from core.crafting_env import (
    BatchCraftingEnvV2, I_SWORD, F_BRIDGE, F_ENEMY_DEFEATED,
    GRID_LIMIT, NUM_ITEMS, RIVER_X_MIN, RIVER_X_MAX,
    BRIDGE_Y_MIN, BRIDGE_Y_MAX
)

class HRLCraftingEnv(BatchCraftingEnvV2):
    def __init__(self, n_envs: int = 32, seed=None, zone_aliases=None):
        super().__init__(n_envs, seed, zone_aliases)
        self.enemy_pos = np.copy(self.zones[:, 5])
        self.enemy_health = np.full(self.n_envs, 100.0, dtype=np.float32)
        
    def reset(self):
        obs, info = super().reset()
        self.enemy_pos = np.copy(self.zones[:, 5])
        self.enemy_health = np.full(self.n_envs, 100.0, dtype=np.float32)
        return obs, info

        
    def step(self, actions):
        # 1. Update enemy positions — only chase if an agent is on the enemy's side of the river
        for i in range(self.n_envs):
            if self.inventory[i, F_ENEMY_DEFEATED] > 0:
                continue
            
            bridge_built = self.inventory[i, F_BRIDGE] > 0
            
            # Check if any agent has crossed the river (is on the enemy's side)
            a0_crossed = self.pos[i, 0, 0] > RIVER_X_MAX
            a1_crossed = self.pos[i, 1, 0] > RIVER_X_MAX
            
            # Enemy only starts chasing if bridge is built AND an agent has crossed
            if not (bridge_built and (a0_crossed or a1_crossed)):
                continue
                
            # Distance to agents that have crossed
            d0 = np.linalg.norm(self.pos[i, 0] - self.enemy_pos[i]) if a0_crossed else 999.0
            d1 = np.linalg.norm(self.pos[i, 1] - self.enemy_pos[i]) if a1_crossed else 999.0
            
            # Find closest agent (that has crossed)
            closest_agent = 0 if d0 < d1 else 1
            dist = min(d0, d1)
            
            # Enemy moves 0.5 units towards the closest crossed agent
            if dist > 0.1:
                direction = self.pos[i, closest_agent] - self.enemy_pos[i]
                direction = direction / np.linalg.norm(direction)
                new_pos = self.enemy_pos[i] + direction * 0.5
                
                # Keep in bounds (enemy stays on far side of river)
                new_pos[0] = max(new_pos[0], RIVER_X_MAX + 1)
                new_pos = np.clip(new_pos, 0, GRID_LIMIT)
                self.enemy_pos[i] = new_pos

        # 2. Hide enemy from base env logic to override combat
        orig_enemy_zones = np.copy(self.zones[:, 5])
        self.zones[:, 5] = np.array([999.0, 999.0])
        
        # 3. Base step (handles movement, other crafting, bounds)
        obs, rewards, dones, truncs, infos = super().step(actions)
        
        # Restore enemy zone
        self.zones[:, 5] = orig_enemy_zones
        
        # 4. Dynamic Combat Logic & Coordination Multiplier
        d0_all = np.linalg.norm(self.pos[:, 0] - self.enemy_pos, axis=1)
        d1_all = np.linalg.norm(self.pos[:, 1] - self.enemy_pos, axis=1)
        
        # Agents must have a sword and be within attack radius (< 2.0)
        a0_hit = (d0_all < 2.0) & (self.inventory[:, I_SWORD] >= 1) & (self.inventory[:, F_ENEMY_DEFEATED] == 0)
        a1_hit = (d1_all < 2.0) & (self.inventory[:, I_SWORD] >= 1) & (self.inventory[:, F_ENEMY_DEFEATED] == 0)
        
        # Auto-reset integration: If done or truncated, reset enemy health and pos for those envs
        terminal = dones | truncs
        if np.any(terminal):
            term_idx = np.where(terminal)[0]
            self.enemy_pos[term_idx] = self.zones[term_idx, 5]
            self.enemy_health[term_idx] = 100.0

        # Apply damage ONLY for non-terminal environments (to prevent hitting the enemy on the exact frame the env resets)
        valid = ~terminal
        both_hit = a0_hit & a1_hit & valid
        only_a0 = a0_hit & ~a1_hit & valid
        only_a1 = a1_hit & ~a0_hit & valid
        
        self.enemy_health[both_hit] -= 50.0
        self.enemy_health[only_a0] -= 10.0
        self.enemy_health[only_a1] -= 10.0
        
        # Defeat check
        just_defeated = (self.enemy_health <= 0) & (self.inventory[:, F_ENEMY_DEFEATED] == 0) & valid
        self.inventory[just_defeated, F_ENEMY_DEFEATED] = 1
        
        # Give the exact same extrinsic reward for defeating the enemy as the base env (+10.0)
        rewards[just_defeated, :] += 10.0
        
        return obs, rewards, dones, truncs, infos
