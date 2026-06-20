import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs_dir = "C:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3/runs"
for run_path in sorted(glob.glob(os.path.join(runs_dir, "*"))):
    if not os.path.isdir(run_path):
        continue
    event_files = [f for f in os.listdir(run_path) if "tfevents" in f]
    if not event_files:
        continue
    path = os.path.join(run_path, event_files[0])
    try:
        ea = EventAccumulator(path)
        ea.Reload()
        tags = ea.Tags()['scalars']
        if 'Episodes/Success_Rate' in tags:
            events = ea.Scalars('Episodes/Success_Rate')
            max_val = max(e.value for e in events)
            print(f"{os.path.basename(run_path)}: Max Success Rate = {max_val:.2%}, Steps = {len(events)}")
    except Exception as e:
        pass
