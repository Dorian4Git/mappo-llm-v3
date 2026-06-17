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

import json
import numpy as np

OPTION_NAMES = [
    "COLLECT_WOOD", "COLLECT_STONE", "CRAFT_PICKAXE", 
    "MINE_IRON", "CRAFT_SWORD", "BUILD_BRIDGE", "FIGHT_ENEMY"
]
NUM_OPTIONS = len(OPTION_NAMES)

class OptionController:
    def __init__(self, n_envs: int = 128):
        self.n_envs = n_envs
        # Default starting options
        self.current_options = {
            "agent_0": "COLLECT_WOOD", 
            "agent_1": "COLLECT_STONE"
        }

    def get_option_embeddings(self) -> np.ndarray:
        """Returns [n_envs, 2, NUM_OPTIONS] one-hot embeddings for the NN."""
        embs = np.zeros((self.n_envs, 2, NUM_OPTIONS), dtype=np.float32)
        
        a0_opt = self.current_options["agent_0"]
        a1_opt = self.current_options["agent_1"]
        
        a0_idx = OPTION_NAMES.index(a0_opt) if a0_opt in OPTION_NAMES else 0
        a1_idx = OPTION_NAMES.index(a1_opt) if a1_opt in OPTION_NAMES else 0
        
        embs[:, 0, a0_idx] = 1.0
        embs[:, 1, a1_idx] = 1.0
        return embs

    def update_options_from_llm(self, llm_json_str):
        if llm_json_str is None:
            return False
            
        try:
            # Clean Markdown if LLM returned json block
            clean_str = llm_json_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            
            # Only update if valid options are present
            if data.get("agent_0_option") and data.get("agent_1_option"):
                self.current_options["agent_0"] = data.get("agent_0_option")
                self.current_options["agent_1"] = data.get("agent_1_option")
                
                # Print the LLM's internal reasoning to your console for debugging
                print(f"[Orchestrator Logic] {data.get('dag_check', 'No reasoning provided')}")
                print(f"[Assigned] A0: {self.current_options['agent_0']} | A1: {self.current_options['agent_1']}")
                return True
            return False
            
        except json.JSONDecodeError:
            print("[Warning] LLM JSON parsing failed. Retaining previous options.")
            return False

    def get_active_option(self, agent_id):
        return self.current_options.get(f"agent_{agent_id}")
