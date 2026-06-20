import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def print_tags(log_dir):
    event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]
    if not event_files:
        print(f"No tfevents in {log_dir}")
        return
    path = os.path.join(log_dir, event_files[0])
    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags()['scalars']
    print(f"Tags for {os.path.basename(log_dir)}: {tags}")

runs_dir = "C:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3/runs"
lora_run = os.path.join(runs_dir, "v3_HRL_Std_E128_s42_20260619-202112")
base_run = os.path.join(runs_dir, "v3_HRL_Std_NoLoRA_E128_s42_20260620-092622")

print_tags(lora_run)
print_tags(base_run)
