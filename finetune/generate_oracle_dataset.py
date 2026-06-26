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
    "MINE_IRON", "CRAFT_SWORD", "CRAFT_ARMOR", "BUILD_BRIDGE", "FIGHT_ENEMY", "COLLECT_GOLD", "IDLE"
]

def determine_options(inv: dict) -> tuple[str, str, str]:
    """
    Given an inventory state, determine the correct (a0_option, a1_option, reasoning).
    Strictly adhere to DAG constraints.
    """
    w, s, p, i, sw, a = inv['wood'], inv['stone'], inv['pickaxe'], inv['iron'], inv['sword'], inv['armor']
    b, e, g = inv.get('bridge', 0), inv.get('enemy', 0), inv.get('gold', 0)
    
    if p == 0:
        if w >= 1 and s >= 1:
            return "CRAFT_PICKAXE", "CRAFT_PICKAXE", "Have Wood and Stone. Both agents craft Pickaxe at workbench."
        elif w >= 1 and b == 0:
            return "BUILD_BRIDGE", "COLLECT_STONE", "Agent 0 builds bridge early, Agent 1 gets stone."
        elif w >= 1 and b == 1:
            return "CRAFT_PICKAXE", "COLLECT_STONE", "Agent 0 waits, Agent 1 gets stone."
        elif s >= 1:
            return "COLLECT_WOOD", "CRAFT_PICKAXE", "Agent 0 gets wood, Agent 1 waits at workbench."
        else:
            return "COLLECT_WOOD", "COLLECT_STONE", "Need pickaxe. Agent 0 gets wood, Agent 1 gets stone."
    else:
        # Determine A1 (Miner/Fighter) option
        if i == 0 and sw == 0 and a == 0:
            a1_opt = "MINE_IRON"
        elif sw == 0 and i >= 1:
            a1_opt = "CRAFT_SWORD"
        elif a == 0 and i >= 1:
            a1_opt = "CRAFT_ARMOR"
        elif i == 0 and (sw == 0 or a == 0):
            a1_opt = "MINE_IRON"
        elif sw >= 1 and a >= 1 and b >= 1 and e == 0:
            a1_opt = "FIGHT_ENEMY"
        elif sw >= 1 and a >= 1 and b == 0 and e == 0:
            a1_opt = "IDLE"
        elif e >= 1 and g == 0:
            a1_opt = "COLLECT_GOLD"
        else:
            a1_opt = "IDLE" # Fallback
            
        # Determine A0 (Lumberjack/Builder) option
        if b == 0 and w == 0:
            a0_opt = "COLLECT_WOOD"
        elif b == 0 and w >= 1:
            a0_opt = "BUILD_BRIDGE"
        else:
            a0_opt = "IDLE" # A0 waits.
            
        return a0_opt, a1_opt, "A0 handles bridge, A1 handles mining and combat."


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
            for bridge in range(2):
                states.append({
                    "wood": wood, "stone": stone, "iron": 0,
                    "pickaxe": 0, "sword": 0, "armor": 0,
                    "gold": 0, "bridge": bridge, "enemy": 0
                })
    
    # Phase 2: Have pickaxe, mining iron
    for wood in range(3):
        for iron in range(3):  # 0, 1, 2
            for bridge in range(2):
                states.append({
                    "wood": wood, "stone": 0, "iron": iron,
                    "pickaxe": 1, "sword": 0, "armor": 0,
                    "gold": 0, "bridge": bridge, "enemy": 0
                })
    
    # Phase 3: Crafting sword and armor (need 2 iron total)
    for wood in range(3):
        for iron in range(3):
            for sword in range(2):
                for armor in range(2):
                    for bridge in range(2):
                        # Skip impossible: can't have sword/armor without having used iron
                        if sword + armor > 2:
                            continue
                        states.append({
                            "wood": wood, "stone": 0, "iron": iron,
                            "pickaxe": 1, "sword": sword, "armor": armor,
                            "gold": 0, "bridge": bridge, "enemy": 0
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
