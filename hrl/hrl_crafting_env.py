import numpy as np
from core.crafting_env import BatchCraftingEnvV2, I_SWORD, F_ENEMY_DEFEATED, GRID_LIMIT, NUM_ITEMS

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
        # 1. Update enemy positions (Aggro / Chasing)
        for i in range(self.n_envs):
            if self.inventory[i, F_ENEMY_DEFEATED] > 0:
                continue
                
            # Distance to both agents
            d0 = np.linalg.norm(self.pos[i, 0] - self.enemy_pos[i])
            d1 = np.linalg.norm(self.pos[i, 1] - self.enemy_pos[i])
            
            # Find closest agent
            closest_agent = 0 if d0 < d1 else 1
            dist = min(d0, d1)
            
            # Enemy moves 0.5 units towards the closest agent
            if dist > 0.1:
                direction = self.pos[i, closest_agent] - self.enemy_pos[i]
                direction = direction / np.linalg.norm(direction)
                self.enemy_pos[i] += direction * 0.5
                
                # Keep in bounds
                self.enemy_pos[i] = np.clip(self.enemy_pos[i], 0, GRID_LIMIT)

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
        
        both_hit = a0_hit & a1_hit
        only_a0 = a0_hit & ~a1_hit
        only_a1 = a1_hit & ~a0_hit
        
        # Apply damage
        self.enemy_health[both_hit] -= 50.0
        self.enemy_health[only_a0] -= 10.0
        self.enemy_health[only_a1] -= 10.0
        
        # Defeat check
        just_defeated = (self.enemy_health <= 0) & (self.inventory[:, F_ENEMY_DEFEATED] == 0)
        self.inventory[just_defeated, F_ENEMY_DEFEATED] = 1
        
        # Give the exact same extrinsic reward for defeating the enemy as the base env (+10.0)
        rewards[just_defeated, :] += 10.0
        
        # Auto-reset integration: If done, reset enemy health and pos for those envs
        if np.any(dones):
            done_idx = np.where(dones)[0]
            self.enemy_pos[done_idx] = self.zones[done_idx, 5]
            self.enemy_health[done_idx] = 100.0
            
        return obs, rewards, dones, truncs, infos
