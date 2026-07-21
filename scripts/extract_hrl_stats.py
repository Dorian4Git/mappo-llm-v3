"""Extract detailed statistics from TensorBoard event files for the 3 HRL runs."""
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

runs = {
    'No-LoRA':     'runs/v3_HRL_Std_Ollama_E128_s42_20260720-085701',
    'QLoRA+Refl':  'runs/v3_HRL_Std_LoRA_E128_s42_20260720-190702',
    'QLoRA-NoRefl':'runs/v3_HRL_Std_LoRA_E128_s42_20260721-094612',
}

tags_of_interest = [
    'Rewards/Avg_Env_Reward',
    'TD_Error/Abs_Mean',
    'Subtasks/Wood_Pct',
    'Subtasks/Stone_Pct',
    'Subtasks/Pickaxe_Pct',
    'Subtasks/Iron_Pct',
    'Subtasks/Sword_Pct',
    'Subtasks/Armor_Pct',
    'Subtasks/Bridge_Pct',
    'Subtasks/Enemy_Pct',
    'Subtasks/Gold_Pct',
    'Subtasks/GameOver_Pct',
    'HRL/Option_Staleness_Resets',
    'HRL/LLM_Queries_Dispatched',
]

for name, path in runs.items():
    print(f"\n{'='*60}")
    print(f"RUN: {name}")
    print(f"{'='*60}")
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    available = ea.Tags().get('scalars', [])
    
    for tag in tags_of_interest:
        if tag not in available:
            print(f"  {tag}: NOT FOUND")
            continue
        events = ea.Scalars(tag)
        vals = np.array([e.value for e in events])
        steps = np.array([e.step for e in events])
        
        # For subtask pct, find first step where value > 0
        if '_Pct' in tag:
            # Scale if in [0,1]
            if vals.max() <= 1.05 and vals.max() > 0:
                vals = vals * 100.0
            nonzero = np.where(vals > 0)[0]
            first_step = steps[nonzero[0]] if len(nonzero) > 0 else -1
            # Final 100 updates average
            final_avg = vals[-100:].mean() if len(vals) >= 100 else vals.mean()
            peak = vals.max()
            print(f"  {tag}: first_nonzero_step={first_step}, final_avg={final_avg:.2f}%, peak={peak:.2f}%")
        elif 'Reward' in tag:
            # Find peak and final average
            final_avg = vals[-100:].mean() if len(vals) >= 100 else vals.mean()
            peak = vals.max()
            peak_step = steps[vals.argmax()]
            print(f"  {tag}: final_avg={final_avg:.4f}, peak={peak:.4f} @step={peak_step}")
        elif 'TD_Error' in tag:
            final_avg = vals[-100:].mean() if len(vals) >= 100 else vals.mean()
            overall_std = vals.std()
            print(f"  {tag}: mean={vals.mean():.4f}, std={overall_std:.4f}, final_avg={final_avg:.4f}")
        else:
            print(f"  {tag}: total={vals.sum():.0f}, mean={vals.mean():.4f}")
