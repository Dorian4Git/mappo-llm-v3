import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs_dir = "C:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3/runs"
runs = [
    "v3_HRL_Std_E128_s42_20260619-202112",
    "v3_HRL_Std_E128_s42_20260620-091111",
    "v3_HRL_Std_NoLoRA_E128_s42_20260620-092517",
    "v3_HRL_Std_NoLoRA_E128_s42_20260620-092622"
]

for r in runs:
    path = os.path.join(runs_dir, r)
    if not os.path.exists(path):
        continue
    event_files = [f for f in os.listdir(path) if "tfevents" in f]
    if not event_files:
        continue
    ea = EventAccumulator(os.path.join(path, event_files[0]))
    ea.Reload()
    if 'Rewards/Avg_Env_Reward' in ea.Tags()['scalars']:
        events = ea.Scalars('Rewards/Avg_Env_Reward')
        print(f"Run {r}: {len(events)} steps logged. Max step: {events[-1].step}")
    else:
        print(f"Run {r}: No reward data")
