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
    "MINE_IRON", "CRAFT_SWORD", "CRAFT_ARMOR", "BUILD_BRIDGE", "FIGHT_ENEMY", "COLLECT_GOLD"
]
NUM_OPTIONS = len(OPTION_NAMES)

class OptionController:
    def __init__(self, n_envs: int = 128):
        self.n_envs = n_envs
        # Track options for BOTH agents separately
        self._active_options_a0 = np.full(n_envs, 0, dtype=np.int32)
        self._active_options_a1 = np.full(n_envs, 1, dtype=np.int32)
        
        # Track pending status PER environment
        self.llm_pending = np.zeros(n_envs, dtype=bool)
        self.cooldown_counter = np.zeros(n_envs, dtype=int)

    def set_pending(self, env_indices, status: bool):
        """Locks or unlocks the LLM query status for specific environments."""
        self.llm_pending[env_indices] = status

    def update_options_from_batch(self, batch_results, env_indices):
        """Parses a batch of LLM responses and updates specific environments.
        Returns a list of dictionaries containing the parsed options for logging.
        """
        parsed_results = []
        for env_idx, llm_json_str in zip(env_indices, batch_results):
            parsed_data = {"agent_0_option": None, "agent_1_option": None}
            if llm_json_str is not None:
                try:
                    clean_str = llm_json_str.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_str)
                    
                    if data.get("agent_0_option") and data.get("agent_1_option"):
                        a0_opt = data.get("agent_0_option")
                        a1_opt = data.get("agent_1_option")
                        
                        self._active_options_a0[env_idx] = OPTION_NAMES.index(a0_opt) if a0_opt in OPTION_NAMES else 0
                        self._active_options_a1[env_idx] = OPTION_NAMES.index(a1_opt) if a1_opt in OPTION_NAMES else 0
                        
                        parsed_data["agent_0_option"] = a0_opt
                        parsed_data["agent_1_option"] = a1_opt
                except json.JSONDecodeError:
                    pass
            parsed_results.append(parsed_data)
        return parsed_results

    def get_option_embeddings(self) -> np.ndarray:
        """Returns [n_envs, 2, NUM_OPTIONS] one-hot embeddings for the NN."""
        embs = np.zeros((self.n_envs, 2, NUM_OPTIONS), dtype=np.float32)
        
        # using advanced indexing
        env_idx = np.arange(self.n_envs)
        embs[env_idx, 0, self._active_options_a0] = 1.0
        embs[env_idx, 1, self._active_options_a1] = 1.0
        return embs

    def update_options_from_llm(self, llm_json_str, env_indices=None):
        if llm_json_str is None:
            return False
            
        try:
            # Clean Markdown if LLM returned json block
            clean_str = llm_json_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            
            # Only update if valid options are present
            if data.get("agent_0_option") and data.get("agent_1_option"):
                a0_opt = data.get("agent_0_option")
                a1_opt = data.get("agent_1_option")
                
                a0_idx = OPTION_NAMES.index(a0_opt) if a0_opt in OPTION_NAMES else 0
                a1_idx = OPTION_NAMES.index(a1_opt) if a1_opt in OPTION_NAMES else 0
                
                # Target specific environments to prevent cross-contamination
                if env_indices is None:
                    self._active_options_a0[:] = a0_idx
                    self._active_options_a1[:] = a1_idx
                else:
                    self._active_options_a0[env_indices] = a0_idx
                    self._active_options_a1[env_indices] = a1_idx
                    
                print(f"[Orchestrator Logic] {data.get('dag_check', 'No reasoning provided')}")
                print(f"[Assigned Envs {env_indices}] A0: {OPTION_NAMES[a0_idx]} | A1: {OPTION_NAMES[a1_idx]}")
                return True
            return False
            
        except json.JSONDecodeError:
            print("[Warning] LLM JSON parsing failed. Retaining previous options.")
            return False

    def get_active_option(self, agent_id, env_id=None):
        """Returns an array of strings if env_id is None, else single string."""
        if agent_id == 0:
            if env_id is None:
                return np.array([OPTION_NAMES[idx] for idx in self._active_options_a0])
            return OPTION_NAMES[self._active_options_a0[env_id]]
        else:
            if env_id is None:
                return np.array([OPTION_NAMES[idx] for idx in self._active_options_a1])
            return OPTION_NAMES[self._active_options_a1[env_id]]
