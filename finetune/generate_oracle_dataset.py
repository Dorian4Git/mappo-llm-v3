"""
generate_oracle_dataset.py — Deterministic Oracle Dataset for QLoRA Fine-Tuning
=================================================================================

Generates a high-quality training dataset by enumerating all meaningful
inventory states along the crafting DAG and applying hardcoded logic to
determine the correct agent assignments.

The DAG (from crafting_env.py):
    Wood (A0, max 2) + Stone (A1, max 1) → Pickaxe (1W+1S at workbench)
    Pickaxe → Iron (A1, max 2) → Sword (A1, 1 Iron at workbench) + Armor (A1, 1 Iron at workbench)
    Wood → Bridge (A0, 1W)
    Bridge + Sword + Armor → Fight Enemy (A1) → Gold (A1)

Valid OPTION_NAMES (must match option_controller.py exactly):
    COLLECT_WOOD, COLLECT_STONE, CRAFT_PICKAXE, MINE_IRON,
    CRAFT_SWORD, CRAFT_ARMOR, BUILD_BRIDGE, FIGHT_ENEMY

Usage:
    python -m finetune.generate_oracle_dataset
    python -m finetune.generate_oracle_dataset --output data/datasets/raw/oracle_dataset.jsonl
"""

import json
import os
import itertools
import random
import argparse
from llm.prompt_builder import PromptBuilder


# Valid options — must match option_controller.py
VALID_OPTIONS = [
    "COLLECT_WOOD", "COLLECT_STONE", "CRAFT_PICKAXE",
    "MINE_IRON", "CRAFT_SWORD", "CRAFT_ARMOR", "BUILD_BRIDGE", "FIGHT_ENEMY"
]

def determine_options(inv: dict) -> tuple[str, str, str]:
    """
    Given an inventory state, determine the correct (a0_option, a1_option, reasoning).
    
    Returns the optimal assignment based on the DAG dependencies.
    Agent 0 (Lumberjack): COLLECT_WOOD, CRAFT_PICKAXE, BUILD_BRIDGE
    Agent 1 (Miner): COLLECT_STONE, CRAFT_PICKAXE, MINE_IRON, CRAFT_SWORD, CRAFT_ARMOR, FIGHT_ENEMY
    
    Key constraints from crafting_env.py:
    - Wood: A0 only, max 2 total (1 for pickaxe, 1 for bridge)
    - Stone: A1 only, max 1 total
    - Iron: A1 only, needs pickaxe, max 2 total (1 for sword, 1 for armor)  
    - Pickaxe: either agent at workbench, costs 1W+1S
    - Sword: A1 at workbench, costs 1 Iron
    - Armor: A1 at workbench, costs 1 Iron
    - Bridge: A0, costs 1W
    - Enemy: A1, needs bridge + sword + armor
    - Gold: A1, needs enemy defeated
    """
    wood = inv["wood"]
    stone = inv["stone"]
    iron = inv["iron"]
    pickaxe = inv["pickaxe"]
    sword = inv["sword"]
    armor = inv["armor"]
    bridge = inv["bridge"]
    enemy = inv["enemy"]
    
    # === PHASE 6: Enemy defeated, go for gold ===
    if enemy >= 1:
        # A1 fights/collects gold. A0 has nothing productive left.
        return "COLLECT_WOOD", "FIGHT_ENEMY", "Enemy defeated. Agent 1 goes to collect Gold while Agent 0 stays productive."
    
    # === PHASE 5: Have sword + armor + bridge, fight enemy ===
    if sword >= 1 and armor >= 1 and bridge >= 1:
        return "COLLECT_WOOD", "FIGHT_ENEMY", "We have Sword, Armor, and Bridge. Agent 1 fights the Enemy."

    # === PHASE 4: Need bridge and/or combat gear ===
    # Determine what's still needed
    need_sword = sword < 1
    need_armor = armor < 1
    need_bridge = bridge < 1
    
    # Sub-case: Have all combat gear, just need bridge
    if not need_sword and not need_armor and need_bridge:
        if wood >= 1:
            return "BUILD_BRIDGE", "FIGHT_ENEMY", "Sword and Armor ready. Agent 0 builds Bridge with available Wood."
        else:
            return "COLLECT_WOOD", "FIGHT_ENEMY", "Need Wood for Bridge. Agent 0 collects Wood."
    
    # Sub-case: Need to craft sword or armor (need iron)
    if need_sword and iron >= 1:
        # A1 crafts sword, A0 works on bridge prep
        if need_bridge and wood < 1:
            return "COLLECT_WOOD", "CRAFT_SWORD", "Have Iron. Agent 1 crafts Sword. Agent 0 collects Wood for Bridge."
        elif need_bridge and wood >= 1:
            return "BUILD_BRIDGE", "CRAFT_SWORD", "Have Iron and Wood. Agent 1 crafts Sword. Agent 0 builds Bridge."
        else:
            return "COLLECT_WOOD", "CRAFT_SWORD", "Have Iron. Agent 1 crafts Sword."
    
    if need_armor and iron >= 1:
        if need_bridge and wood < 1:
            return "COLLECT_WOOD", "CRAFT_ARMOR", "Have Iron. Agent 1 crafts Armor. Agent 0 collects Wood for Bridge."
        elif need_bridge and wood >= 1:
            return "BUILD_BRIDGE", "CRAFT_ARMOR", "Have Iron and Wood. Agent 1 crafts Armor. Agent 0 builds Bridge."
        else:
            return "COLLECT_WOOD", "CRAFT_ARMOR", "Have Iron. Agent 1 crafts Armor."
    
    # Sub-case: Need iron (have pickaxe but no iron)
    if (need_sword or need_armor) and pickaxe >= 1 and iron < 1:
        if need_bridge and wood < 1:
            return "COLLECT_WOOD", "MINE_IRON", "Have Pickaxe. Agent 1 mines Iron. Agent 0 collects Wood for Bridge."
        elif need_bridge and wood >= 1:
            return "BUILD_BRIDGE", "MINE_IRON", "Have Pickaxe and Wood. Agent 1 mines Iron. Agent 0 builds Bridge."
        else:
            return "COLLECT_WOOD", "MINE_IRON", "Have Pickaxe. Agent 1 mines Iron."
    
    # === PHASE 3: Need pickaxe ===
    if pickaxe < 1:
        if wood >= 1 and stone >= 1:
            return "CRAFT_PICKAXE", "CRAFT_PICKAXE", "Have Wood and Stone. Both agents craft Pickaxe at workbench."
        elif wood < 1 and stone < 1:
            return "COLLECT_WOOD", "COLLECT_STONE", "Missing Wood and Stone. Agents collect resources in parallel."
        elif wood < 1:
            return "COLLECT_WOOD", "COLLECT_STONE", "Missing Wood. Agent 0 collects Wood, Agent 1 waits at Stone zone."
        else:  # stone < 1
            return "COLLECT_WOOD", "COLLECT_STONE", "Missing Stone. Agent 1 collects Stone, Agent 0 collects more Wood."
    
    # === PHASE 2: Have pickaxe, need iron ===
    if pickaxe >= 1 and iron < 1:
        if wood < 1 and need_bridge:
            return "COLLECT_WOOD", "MINE_IRON", "Have Pickaxe. Agent 1 mines Iron. Agent 0 collects Wood."
        else:
            return "COLLECT_WOOD", "MINE_IRON", "Have Pickaxe. Agent 1 mines Iron."
    
    # === Fallback: collect resources ===
    return "COLLECT_WOOD", "COLLECT_STONE", "Collecting basic resources to progress the DAG."


def generate_inventory_states():
    """
    Generate all meaningful inventory states along the DAG progression.
    Instead of brute-force enumeration, we generate states that actually
    occur during gameplay.
    """
    states = []
    
    # Phase 1: Early game - collecting wood and stone
    for wood in range(3):  # 0, 1, 2
        for stone in range(2):  # 0, 1
            states.append({
                "wood": wood, "stone": stone, "iron": 0,
                "pickaxe": 0, "sword": 0, "armor": 0,
                "gold": 0, "bridge": 0, "enemy": 0
            })
    
    # Phase 2: Have pickaxe, mining iron
    for wood in range(3):
        for iron in range(3):  # 0, 1, 2
            states.append({
                "wood": wood, "stone": 0, "iron": iron,
                "pickaxe": 1, "sword": 0, "armor": 0,
                "gold": 0, "bridge": 0, "enemy": 0
            })
    
    # Phase 3: Crafting sword and armor (need 2 iron total)
    for wood in range(3):
        for iron in range(3):
            for sword in range(2):
                for armor in range(2):
                    # Skip impossible: can't have sword/armor without having used iron
                    if sword + armor > 2:
                        continue
                    states.append({
                        "wood": wood, "stone": 0, "iron": iron,
                        "pickaxe": 1, "sword": sword, "armor": armor,
                        "gold": 0, "bridge": 0, "enemy": 0
                    })
    
    # Phase 4: Building bridge (need wood)
    for wood in range(3):
        for sword in range(2):
            for armor in range(2):
                for bridge in range(2):
                    states.append({
                        "wood": wood, "stone": 0, "iron": 0,
                        "pickaxe": 1, "sword": sword, "armor": armor,
                        "gold": 0, "bridge": bridge, "enemy": 0
                    })
    
    # Phase 5: Fighting enemy
    for sword in range(2):
        for armor in range(2):
            states.append({
                "wood": 0, "stone": 0, "iron": 0,
                "pickaxe": 1, "sword": sword, "armor": armor,
                "gold": 0, "bridge": 1, "enemy": 0
            })
    
    # Phase 6: Enemy defeated, getting gold
    states.append({
        "wood": 0, "stone": 0, "iron": 0,
        "pickaxe": 1, "sword": 1, "armor": 1,
        "gold": 0, "bridge": 1, "enemy": 1
    })
    
    # Deduplicate
    unique = []
    seen = set()
    for s in states:
        key = tuple(sorted(s.items()))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    
    return unique


def generate_dataset(output_path: str, target_size: int = 1200):
    """Generate the oracle dataset."""
    prompt_builder = PromptBuilder()
    
    base_states = generate_inventory_states()
    print(f"Generated {len(base_states)} unique inventory states")
    
    examples = []
    
    # Generate examples for each state with different agent statuses
    agent_statuses = [
        ("Finished", "Finished"),
        ("Finished", "Working on COLLECT_STONE"),
        ("Working on COLLECT_WOOD", "Finished"),
        ("Finished", "Working on MINE_IRON"),
        ("Working on BUILD_BRIDGE", "Finished"),
        ("Working on COLLECT_WOOD", "Working on COLLECT_STONE"),
        ("Finished", "Working on CRAFT_SWORD"),
        ("Working on CRAFT_PICKAXE", "Working on CRAFT_PICKAXE"),
        ("Finished", "Working on CRAFT_ARMOR"),
        ("Working on BUILD_BRIDGE", "Working on FIGHT_ENEMY"),
    ]
    
    for inv in base_states:
        for a0_status, a1_status in agent_statuses:
            a0_opt, a1_opt, reasoning = determine_options(inv)
            
            # Validate options
            assert a0_opt in VALID_OPTIONS, f"Invalid A0 option: {a0_opt}"
            assert a1_opt in VALID_OPTIONS, f"Invalid A1 option: {a1_opt}"
            
            prompt = prompt_builder.build_hrl_prompt(inv, a0_status, a1_status)
            
            completion = json.dumps({
                "dag_check": reasoning,
                "agent_0_option": a0_opt,
                "agent_1_option": a1_opt
            }, indent=2)
            
            examples.append({
                "prompt": prompt,
                "completion": completion
            })
    
    # Shuffle and trim to target size if needed
    random.seed(42)
    random.shuffle(examples)
    
    if len(examples) > target_size:
        examples = examples[:target_size]
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Generated {len(examples)} examples -> {output_path}")
    
    # Validation: scan all completions
    option_counts = {}
    for ex in examples:
        data = json.loads(ex["completion"])
        for key in ["agent_0_option", "agent_1_option"]:
            opt = data[key]
            option_counts[opt] = option_counts.get(opt, 0) + 1
    
    print("\nOption distribution:")
    for opt, count in sorted(option_counts.items(), key=lambda x: -x[1]):
        valid_marker = "[OK]" if opt in VALID_OPTIONS else "[INVALID]"
        print(f"  {opt}: {count} {valid_marker}")
    
    invalid = [opt for opt in option_counts if opt not in VALID_OPTIONS]
    if invalid:
        print(f"\n[WARNING] INVALID OPTIONS FOUND: {invalid}")
        return False
    else:
        print(f"\n[OK] All {len(examples)} examples use only valid options!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate oracle dataset for QLoRA fine-tuning")
    parser.add_argument("--output", type=str, default="data/datasets/raw/oracle_dataset.jsonl")
    parser.add_argument("--target-size", type=int, default=1200)
    args = parser.parse_args()
    
    generate_dataset(args.output, args.target_size)


if __name__ == "__main__":
    main()
