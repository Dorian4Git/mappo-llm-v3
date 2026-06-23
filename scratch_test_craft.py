import numpy as np
import sys
sys.path.append("c:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3")
from core.crafting_env import BatchCraftingEnvV2, I_WOOD, I_STONE, I_PICKAXE

def main():
    env = BatchCraftingEnvV2(n_envs=1, seed=42)
    obs, _ = env.reset()
    
    # Cheat: give 2 wood, 1 stone
    env.inventory[0, I_WOOD] = 2
    env.inventory[0, I_STONE] = 1
    
    # Teleport Agent 1 to Workbench
    wb_x, wb_y = env.zones[0, 2] # Workbench
    env.pos[0, 1] = [wb_x, wb_y]
    
    print(f"Before: Wood={env.inventory[0, I_WOOD]}, Stone={env.inventory[0, I_STONE]}, Pickaxe={env.inventory[0, I_PICKAXE]}")
    
    # Press Action 4 (Interact) for Agent 1
    actions = np.array([[0, 4]]) # Agent 0 stays (0=up, doesn't matter), Agent 1 interacts
    
    obs, rewards, dones, truncs, infos = env.step(actions)
    
    print(f"After: Wood={env.inventory[0, I_WOOD]}, Stone={env.inventory[0, I_STONE]}, Pickaxe={env.inventory[0, I_PICKAXE]}")
    print(f"Rewards: A0={rewards[0, 0]}, A1={rewards[0, 1]}")

if __name__ == "__main__":
    main()
