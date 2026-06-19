import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.async_bridge import LLMBridge

def test():
    adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models", "qlora_adapter"))
    bridge = LLMBridge(backend="huggingface_peft", model_name=adapter_path)
    bridge.swap_model(adapter_path, backend="huggingface_peft")
    
    prompt = """You are the high-level Cognitive Orchestrator for two MARL agents.
    Strict Dependency DAG: (Wood+Stone) -> Pickaxe -> Iron -> (Sword+Armor) -> Bridge -> Enemy -> Gold.
### CURRENT STATE:
    Inventory: {'wood': 0, 'stone': 0, 'iron': 0, 'pickaxe': 1, 'sword': 0, 'armor': 0, 'bridge': 0, 'enemy': 0, 'gold': 0}
    Agent 0 Status: Idle
    Agent 1 Status: Idle

    ### YOUR TASK:
    Assign exactly ONE discrete Option to each agent to progress the DAG.
    Available Options: ["COLLECT_WOOD", "COLLECT_STONE", "CRAFT_PICKAXE", "MINE_IRON", "CRAFT_SWORD", "CRAFT_ARMOR", "BUILD_BRIDGE", "FIGHT_ENEMY"]
    
    *CRITICAL COOPERATION:* To craft (Pickaxe/Sword/Armor) or build (Bridge), BOTH agents must be assigned the EXACT SAME option simultaneously so they meet at the workbench/bridge.

    Respond ONLY with valid JSON exactly matching this schema:
    {
      "dag_check": "<1 sentence reasoning verifying you have the required inventory for the assigned options>",
      "agent_0_option": "<Option>",
      "agent_1_option": "<Option>"
    }"""
    
    print("Testing RAW prompt exactly as formatting...")
    res = bridge.query_sync(prompt)
    print("RESULT:")
    print(res)

if __name__ == "__main__":
    test()
