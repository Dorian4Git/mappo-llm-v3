import os, glob
from tensorboard.backend.event_processing import event_accumulator

runs = glob.glob('runs/v3_HRL_*')
for r in runs:
    try:
        ea = event_accumulator.EventAccumulator(r, size_guidance={'scalars': 0})
        ea.Reload()
        if 'Rewards/Avg_Env_Reward' in ea.Tags().get('scalars', []):
            events = ea.Scalars('Rewards/Avg_Env_Reward')
            if len(events) > 0:
                print(f'{r}: {len(events)} updates (last step: {events[-1].step})')
    except Exception as e:
        print(f"Error reading {r}: {e}")
