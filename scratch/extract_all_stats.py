"""Extract detailed statistics from TensorBoard event files for the 3 HRL runs."""
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = {
    'No-LoRA':     'runs/v3_HRL_Std_Ollama_E128_s42_20260721-220545',
    'QLoRA+Refl':  'runs/v3_HRL_Std_LoRA_E128_s42_20260722-120127',
    'QLoRA-NoRefl':'runs/v3_HRL_Std_LoRA_E128_s42_20260721-094612',
}

subtasks = ['Wood', 'Stone', 'Pickaxe', 'Iron', 'Sword', 'Armor', 'Bridge', 'Enemy', 'Gold']
other_tags = ['Rewards/Avg_Env_Reward', 'TD_Error/Abs_Mean']

for name, path in runs.items():
    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    print(f"=== {name} ({path}) ===")
    print(f"  Available tags ({len(tags)}): {tags[:25]}")
    for task in subtasks:
        tag = f"Subtasks/{task}_Pct"
        try:
            events = [(e.step, e.value) for e in ea.Scalars(tag)]
            first = next((s for s, v in events if v > 0), None)
            last_val = events[-1][1] if events else 0
            max_val = max((v for _, v in events), default=0)
            print(f"  {task}: first={first}, last={last_val*100:.1f}%, max={max_val*100:.1f}%, n_points={len(events)}")
        except Exception as ex:
            print(f"  {task}: ERROR {ex}")
    for tag in other_tags:
        try:
            events = [(e.step, e.value) for e in ea.Scalars(tag)]
            last_val = events[-1][1] if events else 0
            max_val = max((v for _, v in events), default=0)
            min_val = min((v for _, v in events), default=0)
            print(f"  {tag}: last={last_val:.4f}, max={max_val:.4f}, min={min_val:.4f}, n_points={len(events)}")
        except Exception as ex:
            print(f"  {tag}: ERROR {ex}")
    print()
