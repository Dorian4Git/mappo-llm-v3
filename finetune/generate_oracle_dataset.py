import json
import os
import random

def generate_dag_a_examples():
    """Original MAPPO Simulation: (Wood+Stone) -> Pickaxe -> Iron -> (Sword+Armor) -> Bridge -> Enemy -> Gold."""
    examples = []
    
    # Define rules and bounds
    for w in range(3):
        for s in range(3):
            for i in range(4):
                for p in range(2):
                    for sw in range(2):
                        for a in range(2):
                            for b in range(2):
                                for e in range(2):
                                    for g in range(2):
                                        # Skip invalid logical combinations (e.g., have gold but enemy not defeated)
                                        if g > 0 and e == 0: continue
                                        if e > 0 and b == 0: continue
                                        if b > 0 and (sw == 0 or a == 0): continue
                                        
                                        inv = {"wood": w, "stone": s, "iron": i, "pickaxe": p, "sword": sw, "armor": a, "bridge": b, "enemy": e, "gold": g}
                                        
                                        # Oracle Logic
                                        if g > 0:
                                            ans = {"dag_check": "Gold is mined. All objectives complete.", "agent_0_option": "IDLE", "agent_1_option": "IDLE"}
                                        elif e == 0 and b > 0 and sw > 0 and a > 0:
                                            ans = {"dag_check": "We have Bridge, Sword, and Armor. Ready to fight enemy.", "agent_0_option": "FIGHT_ENEMY", "agent_1_option": "FIGHT_ENEMY"}
                                        elif b == 0 and w > 0 and sw > 0 and a > 0:
                                            ans = {"dag_check": "We have Sword and Armor, and enough Wood for the Bridge. Building Bridge.", "agent_0_option": "BUILD_BRIDGE", "agent_1_option": "BUILD_BRIDGE"}
                                        elif a == 0 and i > 0 and sw > 0:
                                            ans = {"dag_check": "We have Sword and Iron. Ready to craft Armor.", "agent_0_option": "CRAFT_ARMOR", "agent_1_option": "CRAFT_ARMOR"}
                                        elif sw == 0 and i > 0:
                                            ans = {"dag_check": "We have Iron. Ready to craft Sword.", "agent_0_option": "CRAFT_SWORD", "agent_1_option": "CRAFT_SWORD"}
                                        elif i < 2 and p > 0:
                                            ans = {"dag_check": "We have Pickaxe but lack sufficient Iron. Mining Iron.", "agent_0_option": "IDLE", "agent_1_option": "MINE_IRON"}
                                        elif p == 0 and w > 0 and s > 0:
                                            ans = {"dag_check": "We have Wood and Stone. Ready to craft Pickaxe.", "agent_0_option": "CRAFT_PICKAXE", "agent_1_option": "CRAFT_PICKAXE"}
                                        else:
                                            ans = {"dag_check": "Missing basic resources. Collecting Wood and Stone.", "agent_0_option": "COLLECT_WOOD" if w == 0 else "IDLE", "agent_1_option": "COLLECT_STONE" if s == 0 else "IDLE"}
                                        
                                        prompt = f"""You are the high-level Cognitive Orchestrator for two MARL agents.
    Strict Dependency DAG: (Wood+Stone) -> Pickaxe -> Iron -> (Sword+Armor) -> Bridge -> Enemy -> Gold.
### CURRENT STATE:
    Inventory: {inv}
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
                                        examples.append({"prompt": prompt, "completion": json.dumps(ans, indent=2)})
    return examples

def generate_dag_b_examples():
    """Boat & Fishing Simulation: Wood -> Boat -> Fish -> Market -> Gold"""
    examples = []
    
    for w in range(4):
        for b in range(2):
            for f in range(3):
                for m in range(2):
                    for g in range(2):
                        if g > 0 and m == 0: continue
                        if m > 0 and f == 0: continue
                        if f > 0 and b == 0: continue
                        
                        inv = {"wood": w, "boat": b, "fish": f, "market": m, "gold": g}
                        
                        if g > 0:
                            ans = {"dag_check": "We have Gold. Mission accomplished.", "agent_0_option": "IDLE", "agent_1_option": "IDLE"}
                        elif f > 0 and m == 0:
                            ans = {"dag_check": "We have Fish. Selling at Market.", "agent_0_option": "SELL_MARKET", "agent_1_option": "SELL_MARKET"}
                        elif b > 0 and f == 0:
                            ans = {"dag_check": "We have a Boat. Going fishing.", "agent_0_option": "IDLE", "agent_1_option": "GO_FISHING"}
                        elif w > 0 and b == 0:
                            ans = {"dag_check": "We have Wood. Building Boat.", "agent_0_option": "BUILD_BOAT", "agent_1_option": "BUILD_BOAT"}
                        else:
                            ans = {"dag_check": "Missing Wood. Chopping Wood.", "agent_0_option": "CHOP_WOOD", "agent_1_option": "CHOP_WOOD"}
                        
                        prompt = f"""You are the high-level Cognitive Orchestrator for two MARL agents.
    Strict Dependency DAG: Wood -> Boat -> Fish -> Market -> Gold.
### CURRENT STATE:
    Inventory: {inv}
    Agent 0 Status: Idle
    Agent 1 Status: Idle

    ### YOUR TASK:
    Assign exactly ONE discrete Option to each agent to progress the DAG.
    Available Options: ["CHOP_WOOD", "BUILD_BOAT", "GO_FISHING", "SELL_MARKET"]

    Respond ONLY with valid JSON exactly matching this schema:
    {{
      "dag_check": "<1 sentence reasoning verifying you have the required inventory for the assigned options>",
      "agent_0_option": "<Option>",
      "agent_1_option": "<Option>"
    }}"""
                        examples.append({"prompt": prompt, "completion": json.dumps(ans, indent=2)})
    return examples

def generate_dag_c_examples():
    """Factory Simulation: Clay -> Furnace -> Iron -> Engine -> Car"""
    examples = []
    
    for c in range(3):
        for f in range(2):
            for i in range(3):
                for e in range(2):
                    for car in range(2):
                        if car > 0 and e == 0: continue
                        if e > 0 and i == 0: continue
                        if i > 0 and f == 0: continue
                        
                        inv = {"clay": c, "furnace": f, "iron": i, "engine": e, "car": car}
                        
                        if car > 0:
                            ans = {"dag_check": "Car assembled. Mission accomplished.", "agent_0_option": "IDLE", "agent_1_option": "IDLE"}
                        elif e > 0 and car == 0:
                            ans = {"dag_check": "We have an Engine. Assembling Car.", "agent_0_option": "ASSEMBLE_CAR", "agent_1_option": "ASSEMBLE_CAR"}
                        elif i > 0 and e == 0:
                            ans = {"dag_check": "We have Iron. Crafting Engine.", "agent_0_option": "CRAFT_ENGINE", "agent_1_option": "CRAFT_ENGINE"}
                        elif f > 0 and i == 0:
                            ans = {"dag_check": "We have a Furnace. Smelting Iron.", "agent_0_option": "IDLE", "agent_1_option": "SMELT_IRON"}
                        elif c > 0 and f == 0:
                            ans = {"dag_check": "We have Clay. Building Furnace.", "agent_0_option": "BUILD_FURNACE", "agent_1_option": "BUILD_FURNACE"}
                        else:
                            ans = {"dag_check": "Missing Clay. Digging Clay.", "agent_0_option": "DIG_CLAY", "agent_1_option": "DIG_CLAY"}
                            
                        prompt = f"""You are the high-level Cognitive Orchestrator for two MARL agents.
    Strict Dependency DAG: Clay -> Furnace -> Iron -> Engine -> Car.
### CURRENT STATE:
    Inventory: {inv}
    Agent 0 Status: Idle
    Agent 1 Status: Idle

    ### YOUR TASK:
    Assign exactly ONE discrete Option to each agent to progress the DAG.
    Available Options: ["DIG_CLAY", "BUILD_FURNACE", "SMELT_IRON", "CRAFT_ENGINE", "ASSEMBLE_CAR"]

    Respond ONLY with valid JSON exactly matching this schema:
    {{
      "dag_check": "<1 sentence reasoning verifying you have the required inventory for the assigned options>",
      "agent_0_option": "<Option>",
      "agent_1_option": "<Option>"
    }}"""
                        examples.append({"prompt": prompt, "completion": json.dumps(ans, indent=2)})
    return examples

def main():
    os.makedirs("data/datasets/raw", exist_ok=True)
    out_path = "data/datasets/raw/oracle_dataset.jsonl"
    
    examples = []
    
    # 1. Generate MAPPO Dataset
    all_examples = []
    
    # 10x artificial scaling
    for _ in range(10):
        all_examples.extend(generate_dag_a_examples())
        all_examples.extend(generate_dag_b_examples())
        all_examples.extend(generate_dag_c_examples())
        
    random.shuffle(all_examples)
    
    print(f"Total Unique Examples: {len(all_examples)}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Wrote oracle data to {out_path}")

if __name__ == "__main__":
    main()
