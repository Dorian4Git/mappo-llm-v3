import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.async_bridge import LLMBridge
from llm.prompt_builder import PromptBuilder

def generate_valid_states():
    # inventory: [wood, stone, iron, pickaxe, sword, armor, gold, bridge, enemy, gameover]
    states = [
        # 0. Initial state
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 1. Partial collection
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # 2. Ready for Pickaxe
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # 3. Pickaxe acquired
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # 4. Iron mined
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0, 0, 0, 0],
        # 5. Sword crafted
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        # 6. Iron mined again
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        # 7. Armor crafted
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        # 8. Wood collected again for Bridge
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        # 9. Bridge built
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
        # 10. Enemy defeated
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    ]
    
    statuses = [
        ("Idle", "Idle"),
        ("Finished", "Finished"),
        ("Working on task", "Working on task"),
    ]
    
    examples = []
    # To enrich the dataset, let's also add some augmented variants where 
    # unrelated items are in inventory (like extra wood)
    enriched_states = list(states)
    for state in states:
        aug_wood = list(state)
        aug_wood[0] += 1
        if aug_wood not in enriched_states:
            enriched_states.append(aug_wood)
            
    for inv in enriched_states:
        for a0_stat, a1_stat in statuses:
            examples.append({
                "inventory": inv,
                "a0_status": a0_stat,
                "a1_status": a1_stat
            })
            
    return examples

def get_oracle_response(inv) -> dict:
    # inv: [wood, stone, iron, pickaxe, sword, armor, gold, bridge, enemy, gameover]
    w, s, i, p, sw, a, g, b, e, _ = inv
    
    if g > 0:
        return {"dag_check": "Gold is mined. All objectives complete.", "agent_0_option": "IDLE", "agent_1_option": "IDLE"}
    if e == 0 and b > 0 and sw > 0 and a > 0:
        return {"dag_check": "We have Bridge, Sword, and Armor. Ready to fight enemy.", "agent_0_option": "FIGHT_ENEMY", "agent_1_option": "FIGHT_ENEMY"}
    if b == 0 and w > 0 and sw > 0 and a > 0:
        return {"dag_check": "We have Sword and Armor, and enough Wood for the Bridge. Building Bridge.", "agent_0_option": "BUILD_BRIDGE", "agent_1_option": "BUILD_BRIDGE"}
    if a == 0 and i > 0 and sw > 0:
        return {"dag_check": "We have Sword and Iron. Ready to craft Armor.", "agent_0_option": "CRAFT_ARMOR", "agent_1_option": "CRAFT_ARMOR"}
    if sw == 0 and i > 0:
        return {"dag_check": "We have Iron. Ready to craft Sword.", "agent_0_option": "CRAFT_SWORD", "agent_1_option": "CRAFT_SWORD"}
    if i < 2 and p > 0:
        return {"dag_check": "We have Pickaxe but lack sufficient Iron. Mining Iron.", "agent_0_option": "IDLE", "agent_1_option": "MINE_IRON"}
    if p == 0 and w > 0 and s > 0:
        return {"dag_check": "We have Wood and Stone. Ready to craft Pickaxe.", "agent_0_option": "CRAFT_PICKAXE", "agent_1_option": "CRAFT_PICKAXE"}
    
    # Base collection
    return {
        "dag_check": "Missing basic resources. Collecting Wood and Stone.", 
        "agent_0_option": "COLLECT_WOOD" if w == 0 else "IDLE", 
        "agent_1_option": "COLLECT_STONE" if s == 0 else "IDLE"
    }

def main():
    os.makedirs("data/datasets/raw", exist_ok=True)
    out_path = "data/datasets/raw/oracle_dataset.jsonl"
    
    prompt_builder = PromptBuilder()
    states = generate_valid_states()
    
    print(f"Generating deterministic oracle responses for {len(states)} states...")
    
    success_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, state in enumerate(states):
            prompt = prompt_builder.build_hrl_prompt(
                state["inventory"], 
                state["a0_status"], 
                state["a1_status"]
            )
            response_dict = get_oracle_response(state["inventory"])
            response_text = json.dumps(response_dict, indent=2)
            
            record = {
                "prompt": prompt,
                "completion": response_text
            }
            f.write(json.dumps(record) + "\n")
            success_count += 1
                
    print(f"\nDone! Generated {success_count} perfect Oracle examples to {out_path}.")

if __name__ == "__main__":
    main()
