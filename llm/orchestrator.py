"""
orchestrator.py — LLM Orchestrator (Inherited from V2 + Async Bridge)
======================================================================

Features:
- Deterministic DAG planner for goals (unchanged from V2)
- LLM Adaptive weights calculation via async bridge
- query_intervention() method for Phase 2 critic-triggered interventions

Targets are enums:
Agent 0: "Wood", "Workbench", "Bridge", "Enemy", "None"
Agent 1: "Stone", "Iron", "Workbench", "Enemy", "Gold", "None"
"""

import json
import time
import numpy as np
import os
from typing import Optional

def apply_dag_guardrails(weights: dict, metrics: dict) -> dict:
    """
    Programmatic Guardrail: Force weights of unreachable DAG tasks to 0.0.
    DAG: (Wood+Stone) -> Pickaxe -> Iron -> (Sword+Armor) -> Bridge -> Defeat Enemy -> Gold
    """
    guarded = dict(weights)
    
    # 1. Iron requires Pickaxe
    if metrics.get('pickaxe', 0.0) < 1.0:
        guarded['w_iron'] = 0.0
        
    # 2. Enemy and Gold shouldn't be targeted until Sword+Armor+Bridge
    # Note: We do NOT zero out w_bridge because Agent 0 needs to target it while waiting!
    if metrics.get('bridge', 0.0) < 1.0 or metrics.get('sword', 0.0) < 1.0 or metrics.get('armor', 0.0) < 1.0:
        guarded['w_enemy'] = 0.0
        guarded['w_gold'] = 0.0
        
    return guarded

R_LLM_SCALE = 0.05

# Targets mapping to Zone Indices in BatchCraftingEnvV2.zones
# 0:wood, 1:stone, 2:workbench, 3:iron, 4:bridge, 5:enemy, 6:gold
TARGET_TO_ZONE = {
    "Wood": 0,
    "Stone": 1,
    "Workbench": 2,
    "Iron": 3,
    "Bridge": 4,
    "Enemy": 5,
    "Gold": 6,
    "None": -1
}

def _greedy_heuristic_planner(w, s, i, p, sw, a, g, b, e, go, subtask_weights=None):
    """
    Greedy planner that assigns the target with the highest (Base Reward * LLM Weight).
    Causes local minima in static baseline but works perfectly with LLM dynamic weights.
    """
    if subtask_weights is None:
        subtask_weights = {}
        
    w_wood = subtask_weights.get('w_wood', 1.0)
    w_stone = subtask_weights.get('w_stone', 1.0)
    w_workbench = subtask_weights.get('w_workbench', 1.0)
    w_iron = subtask_weights.get('w_iron', 1.0)
    w_bridge = subtask_weights.get('w_bridge', 1.0)
    w_enemy = subtask_weights.get('w_enemy', 1.0)
    w_gold = subtask_weights.get('w_gold', 1.0)

    # Base rewards (acting as greedy heuristics for standard dense shaping)
    R_WOOD = 2.0
    R_STONE = 2.0
    R_WORKBENCH = 1.0
    R_IRON = 2.0
    R_BRIDGE = 3.0
    R_ENEMY = 10.0
    R_GOLD = 15.0

    if go > 0 or g > 0:
        return "None", "None"
        
    # --- Agent 0 (Lumberjack) valid tasks ---
    a0_scores = {}
    if w < 2:
        a0_scores["Wood"] = R_WOOD * w_wood
    
    needs_wb_a0 = (p == 0) or (sw == 0) or (a == 0)
    if needs_wb_a0:
        a0_scores["Workbench"] = R_WORKBENCH * w_workbench
        
    if b == 0:
        a0_scores["Bridge"] = R_BRIDGE * w_bridge
    if e == 0:
        a0_scores["Enemy"] = R_ENEMY * w_enemy
        
    a0_target = max(a0_scores.items(), key=lambda x: x[1])[0] if a0_scores else "None"

    # --- Agent 1 (Miner) valid tasks ---
    a1_scores = {}
    if s < 1:
        a1_scores["Stone"] = R_STONE * w_stone
        
    needs_wb_a1 = (p == 0) or (i > 0 and (sw == 0 or a == 0))
    if needs_wb_a1:
        a1_scores["Workbench"] = R_WORKBENCH * w_workbench
        
    if i < 2:
        a1_scores["Iron"] = R_IRON * w_iron
    if e == 0:
        a1_scores["Enemy"] = R_ENEMY * w_enemy
    if g == 0:
        a1_scores["Gold"] = R_GOLD * w_gold
        
    a1_target = max(a1_scores.items(), key=lambda x: x[1])[0] if a1_scores else "None"

    return a0_target, a1_target


def batch_lookup_goals_v2(inventory: np.ndarray, llm_host=None, llm_model=None, log_file=None, subtask_weights=None):
    """
    Args:
        inventory: [n_envs, 10] integer array
        subtask_weights: dict of LLM adaptive weights
    Returns:
        zone_indices: [n_envs, 2] int32 array of target zones (0-6, or -1)
        active: [n_envs, 2] bool array
    """
    n_envs = inventory.shape[0]
    zone_indices = np.full((n_envs, 2), -1, dtype=np.int32)
    active = np.zeros((n_envs, 2), dtype=bool)
    
    for idx in range(n_envs):
        inv = inventory[idx]
        state = (
            int(inv[0]), int(inv[1]), int(inv[2]), int(inv[3]), int(inv[4]),
            int(inv[5]), int(inv[6]), int(inv[7]), int(inv[8]), int(inv[9])
        )
        goals = _greedy_heuristic_planner(*state, subtask_weights=subtask_weights)
        
        for ai in range(2):
            target_name = goals[ai]
            zone_idx = TARGET_TO_ZONE.get(target_name, -1)
            if zone_idx >= 0:
                zone_indices[idx, ai] = zone_idx
                active[idx, ai] = True
                
    return zone_indices, active


def compute_shaped_reward_batch_v2(
    obs_next: np.ndarray,
    obs_prev: np.ndarray,
    goal_zone_indices: np.ndarray,
    goal_active: np.ndarray,
    env_zones: np.ndarray,
    gamma: float = 0.99,
    subtask_weights: dict = None,
) -> np.ndarray:
    """
    Vectorized potential-based shaped reward dynamically using real-time env zones.
    """
    pos_next = np.stack([
        obs_next[:, 0, 0:2],
        obs_next[:, 0, 2:4],
    ], axis=1)

    pos_prev = np.stack([
        obs_prev[:, 0, 0:2],
        obs_prev[:, 0, 2:4],
    ], axis=1)

    n_envs = pos_next.shape[0]
    env_idx = np.arange(n_envs)[:, np.newaxis]
    safe_zone_indices = np.clip(goal_zone_indices, 0, 6)
    goal_targets = env_zones[env_idx, safe_zone_indices]

    dist_next = np.linalg.norm(pos_next - goal_targets, axis=2)
    dist_prev = np.linalg.norm(pos_prev - goal_targets, axis=2)

    MAX_DIST = 85.0
    phi_next = MAX_DIST - dist_next
    phi_prev = MAX_DIST - dist_prev
    F = (gamma * phi_next - phi_prev) * R_LLM_SCALE * goal_active

    # We do NOT scale F by agent_w anymore.
    # Scaling F by transient LLM weights (especially EMA-dragged weights) destroys PPO's value function stability.
    # The LLM's influence is already perfectly applied via the _greedy_heuristic_planner which sets the goal_targets.
    # Once a goal is selected, we want a consistent, strong shaping gradient to guide the agent there.

    return F


class LLMOrchestratorV2:
    """
    LLM Orchestrator with deterministic DAG planner and adaptive weight querying.

    V3 additions:
    - Integration with LLMBridge for async/retry queries
    - query_intervention() method for critic-triggered interventions
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        host: str = "http://localhost:11434/api/generate",
        log_dir: str = ".",
        bridge=None,
    ):
        self.model_name = model_name
        self.host = host
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        log_path = os.path.join(log_dir, f"adaptive_weights_log_{timestamp}.jsonl")
        self.log_file = open(log_path, "a", encoding="utf-8")

        # V3: Optional async bridge
        self._bridge = bridge

    def batch_lookup_goals_v2_randomized(self, inventory: np.ndarray, subtask_weights: dict = None) -> tuple[np.ndarray, np.ndarray]:
        return batch_lookup_goals_v2(inventory, self.host, self.model_name, None, subtask_weights)

    def query_adaptive_weights(self, curr: dict, delta: dict, prev_weights: dict = None) -> dict:
        if prev_weights is None:
            prev_weights = {
                'w_wood': 1.0, 'w_stone': 1.0, 'w_workbench': 1.0,
                'w_iron': 1.0, 'w_bridge': 1.0, 'w_enemy': 1.0, 'w_gold': 1.0,
            }

        prompt = f"""You are an expert diagnostic AI tuning the reward function for a MARL environment.
Strict Dependency DAG: (Wood+Stone) -> Pickaxe -> Iron -> (Sword+Armor) -> Bridge -> Defeat Enemy -> Gold.

### METRICS (Current Epochs vs Previous):
* Wood Collected: {curr.get('wood', 0):.1f}% (Delta: {delta.get('wood', 0):+.1f}%)
* Pickaxe Crafted: {curr.get('pickaxe', 0):.1f}% (Delta: {delta.get('pickaxe', 0):+.1f}%)
* Iron Mined: {curr.get('iron', 0):.1f}% (Delta: {delta.get('iron', 0):+.1f}%)
* Sword Crafted: {curr.get('sword', 0):.1f}% (Delta: {delta.get('sword', 0):+.1f}%)
* Bridge Built: {curr.get('bridge', 0):.1f}% (Delta: {delta.get('bridge', 0):+.1f}%)
* Enemy Defeats: {curr.get('enemy', 0):.1f}% (Delta: {delta.get('enemy', 0):+.1f}%)
* Gold Mined: {curr.get('gold', 0):.1f}% (Delta: {delta.get('gold', 0):+.1f}%)

### YOUR TASK:
1. Identify the current bottleneck in the DAG based on the deltas.
2. Output a JSON object with a brief "reasoning" string, followed by updated weights (0.0 to 1.0).
3. Increase weights for failing tasks; decrease weights for mastered or over-farmed tasks to prevent reward hacking.

Respond ONLY with valid JSON exactly matching this schema:
{{
  "reasoning": "<1 sentence explaining the bottleneck>",
  "w_wood": <float>,
  "w_stone": <float>,
  "w_workbench": <float>,
  "w_iron": <float>,
  "w_bridge": <float>,
  "w_enemy": <float>,
  "w_gold": <float>
}}"""

        try:
            # V3: Use bridge if available, otherwise direct request
            if self._bridge:
                raw_text = self._bridge.query_sync(prompt)
            else:
                import requests
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": 42
                    }
                }
                response = requests.post(self.host, json=payload, timeout=30)
                raw_text = response.json()["response"]
            
            # Safe JSON parsing
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            raw = json.loads(clean_text)
            
            # Log reasoning
            log_data = {
                "prompt": prompt,
                "metrics": curr,
                "deltas": delta,
                "reasoning": raw.get("reasoning", "No reasoning provided"),
                "raw_weights": raw
            }
            self.log_file.write(json.dumps(log_data) + "\n")
            self.log_file.flush()

            # Parse weights, applying clipping
            keys = ['w_wood', 'w_stone', 'w_workbench', 'w_iron', 'w_bridge', 'w_enemy', 'w_gold']
            parsed_weights = {}
            for k in keys:
                parsed_weights[k] = float(np.clip(raw.get(k, prev_weights.get(k, 1.0)), 0.0, 1.0))
            
            # EMA Smoothing (alpha = 0.2)
            alpha = 0.2
            smoothed_weights = {}
            for k in keys:
                smoothed_weights[k] = alpha * parsed_weights[k] + (1.0 - alpha) * prev_weights.get(k, 1.0)
                
            # Programmatic Guardrails to strictly enforce DAG impossibilities
            smoothed_weights = apply_dag_guardrails(smoothed_weights, curr)
                
            # Normalization (ensure average is 1.0)
            eps = 1e-6
            w_sum = sum(smoothed_weights.values()) + eps
            factor = len(keys) / w_sum
            normalized_weights = {k: v * factor for k, v in smoothed_weights.items()}

            return normalized_weights
            
        except Exception as e:
            print(f"[LLM] Error querying adaptive weights: {e}")
            return prev_weights

    def query_intervention_async(self, prompt: str, callback) -> None:
        """
        Query the LLM for an intervention asynchronously.
        """
        try:
            if self._bridge:
                self._bridge.query_async(prompt, callback=callback)
            else:
                # Fallback to sync if no bridge (e.g. testing)
                import requests
                import threading
                def _run():
                    try:
                        payload = {
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json",
                            "options": {
                                "temperature": 0.0,
                                "top_p": 1.0,
                                "seed": 42
                            }
                        }
                        response = requests.post(self.host, json=payload, timeout=30)
                        raw_text = response.json()["response"]
                        callback(raw_text)
                    except Exception as e:
                        print(f"[LLM] Intervention query failed: {e}")
                        callback(None)
                threading.Thread(target=_run, daemon=True).start()

        except Exception as e:
            print(f"[LLM] Intervention query failed to start: {e}")
            callback(None)

    def query_intervention(self, prompt: str) -> Optional[dict]:
        # Kept for backward compatibility
        try:
            if self._bridge:
                raw_text = self._bridge.query_sync(prompt)
            else:
                import requests
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": 42
                    }
                }
                response = requests.post(self.host, json=payload, timeout=30)
                raw_text = response.json()["response"]

            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)

        except Exception as e:
            print(f"[LLM] Intervention query failed: {e}")
            return None


if __name__ == "__main__":
    print("Deterministic DAG Planner Test")
    inv = np.zeros((1, 10), dtype=np.int32)
    zone_idx, active = batch_lookup_goals_v2(inv)
    print("Targets for [0]*10 :", zone_idx, active)
