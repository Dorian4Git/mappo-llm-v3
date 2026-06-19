import json
import time
import os
import sys

# Ensure the root directory is in sys.path so we can import llm.async_bridge
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.async_bridge import LLMBridge

def test_scenario(bridge: LLMBridge, scenario_name: str, inventory: dict, expected_actions: dict):
    print(f"\n[{scenario_name}]")
    print(f"Inventory: {inventory}")
    
    prompt = f"""You are the high-level Cognitive Orchestrator for two MARL agents.
    Strict Dependency DAG: (Wood+Stone) -> Pickaxe -> Iron -> (Sword+Armor) -> Bridge -> Enemy -> Gold.
### CURRENT STATE:
    Inventory: {inventory}
    Agent 0 Status: Idle
    Agent 1 Status: Idle

    ### YOUR TASK:
    Assign exactly ONE discrete Option to each agent to progress the DAG.
    Available Options: ["COLLECT_WOOD", "COLLECT_STONE", "CRAFT_PICKAXE", "MINE_IRON", "CRAFT_SWORD", "CRAFT_ARMOR", "BUILD_BRIDGE", "FIGHT_ENEMY"]

    Respond ONLY with valid JSON exactly matching this schema:
    {{
      "dag_check": "<1 sentence reasoning verifying you have the required inventory for the assigned options>",
      "agent_0_option": "<Option>",
      "agent_1_option": "<Option>"
    }}"""
    
    start_time = time.time()
    try:
        response_text = bridge.query_sync(prompt, timeout=60)
        generation_time = time.time() - start_time
        
        # Qwen usually puts JSON in a markdown block, or just returns raw JSON
        # Let's clean it up if needed
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
            
        result = json.loads(clean_text)
        
        agent_0_opt = result.get("agent_0_option")
        agent_1_opt = result.get("agent_1_option")
        
        success = (agent_0_opt == expected_actions.get("agent_0_option") and 
                   agent_1_opt == expected_actions.get("agent_1_option"))
        
        print(f"Time: {generation_time:.2f}s")
        print(f"Model Output: {result}")
        if success:
            print("[SUCCESS]: Actions match expected!")
        else:
            print(f"[FAILED]: Expected {expected_actions}")
            
        return success
        
    except Exception as e:
        print(f"[ERROR]: Query failed or invalid JSON: {e}")
        print(f"Raw output was: {response_text if 'response_text' in locals() else 'None'}")
        return False

def main():
    print("Loading Fine-Tuned Model...")
    adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models", "qlora_adapter"))
    
    bridge = LLMBridge(backend="huggingface_peft", model_name=adapter_path)
    bridge.swap_model(adapter_path, backend="huggingface_peft")
    
    print("\nStarting Evaluation...")
    
    scenarios = [
        {
            "name": "Scenario 1: Starting out (No items)",
            "inventory": {"wood": 0, "stone": 0, "iron": 0, "pickaxe": 0, "sword": 0, "armor": 0, "bridge": 0, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "COLLECT_WOOD", "agent_1_option": "COLLECT_STONE"}
        },
        {
            "name": "Scenario 2: Have Wood but no Stone",
            "inventory": {"wood": 1, "stone": 0, "iron": 0, "pickaxe": 0, "sword": 0, "armor": 0, "bridge": 0, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "IDLE", "agent_1_option": "COLLECT_STONE"}
        },
        {
            "name": "Scenario 3: Have materials for Pickaxe",
            "inventory": {"wood": 1, "stone": 1, "iron": 0, "pickaxe": 0, "sword": 0, "armor": 0, "bridge": 0, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "CRAFT_PICKAXE", "agent_1_option": "CRAFT_PICKAXE"}
        },
        {
            "name": "Scenario 4: Have Pickaxe, need Iron Ore",
            "inventory": {"wood": 0, "stone": 0, "iron": 0, "pickaxe": 1, "sword": 0, "armor": 0, "bridge": 0, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "IDLE", "agent_1_option": "MINE_IRON"}
        },
        {
            "name": "Scenario 5: Have Iron Ore, need Wood to smelt",
            "inventory": {"wood": 0, "stone": 0, "iron": 1, "pickaxe": 1, "sword": 0, "armor": 0, "bridge": 0, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "CRAFT_SWORD", "agent_1_option": "CRAFT_SWORD"}
        },
        {
            "name": "Scenario 6: Have Iron Ore and Sword, ready to craft Armor",
            "inventory": {"wood": 0, "stone": 0, "iron": 1, "pickaxe": 1, "sword": 1, "armor": 0, "bridge": 0, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "CRAFT_ARMOR", "agent_1_option": "CRAFT_ARMOR"}
        },
        {
            "name": "Scenario 7: Ready to build Bridge",
            "inventory": {"wood": 1, "stone": 0, "iron": 0, "pickaxe": 1, "sword": 1, "armor": 1, "bridge": 0, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "BUILD_BRIDGE", "agent_1_option": "BUILD_BRIDGE"}
        },
        {
            "name": "Scenario 8: Ready to fight Enemy",
            "inventory": {"wood": 0, "stone": 0, "iron": 0, "pickaxe": 1, "sword": 1, "armor": 1, "bridge": 1, "enemy": 0, "gold": 0},
            "expected": {"agent_0_option": "FIGHT_ENEMY", "agent_1_option": "FIGHT_ENEMY"}
        }
    ]
    
    passed = 0
    for s in scenarios:
        if test_scenario(bridge, s["name"], s["inventory"], s["expected"]):
            passed += 1
            
    print(f"\nEvaluation Complete! Passed {passed}/{len(scenarios)} scenarios.")
    bridge.close()

if __name__ == "__main__":
    main()
