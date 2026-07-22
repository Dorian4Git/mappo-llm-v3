"""Extract HRL-specific tags and generate all comparison data."""
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = {
    'No-LoRA':     'runs/v3_HRL_Std_Ollama_E128_s42_20260721-220545',
    'QLoRA+Refl':  'runs/v3_HRL_Std_LoRA_E128_s42_20260722-120127',
    'QLoRA-NoRefl':'runs/v3_HRL_Std_LoRA_E128_s42_20260721-094612',
}

for name, path in runs.items():
    ea = EventAccumulator(path)
    ea.Reload()
    print(f"=== {name} ===")

    # HRL tags
    for tag in ['HRL/Option_Staleness_Resets', 'HRL/LLM_Queries_Dispatched']:
        try:
            events = [(e.step, e.value) for e in ea.Scalars(tag)]
            total = sum(v for _, v in events)
            print(f"  {tag}: total={total:.0f}, n_points={len(events)}")
        except Exception as ex:
            print(f"  {tag}: ERROR {ex}")

    # Reward details
    for tag in ['Rewards/Avg_Env_Reward', 'Rewards/Avg_Total_Reward', 'Rewards/Avg_Intrinsic_Reward']:
        try:
            events = [(e.step, e.value) for e in ea.Scalars(tag)]
            last100 = events[-100:]
            avg_last100 = sum(v for _, v in last100) / len(last100)
            peak = max(v for _, v in events)
            peak_step = next(s for s, v in events if v == peak)
            print(f"  {tag}: avg_last100={avg_last100:.4f}, peak={peak:.4f} at step {peak_step}")
        except Exception as ex:
            print(f"  {tag}: ERROR {ex}")

    # TD Error details
    try:
        events = [(e.step, e.value) for e in ea.Scalars('TD_Error/Abs_Mean')]
        last100 = events[-100:]
        avg_last100 = sum(v for _, v in last100) / len(last100)
        total_avg = sum(v for _, v in events) / len(events)
        print(f"  TD_Error: avg_all={total_avg:.4f}, avg_last100={avg_last100:.4f}")
    except Exception as ex:
        print(f"  TD_Error: ERROR {ex}")

    print()
