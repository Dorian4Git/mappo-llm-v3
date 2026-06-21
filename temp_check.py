import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

d1 = 'C:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3/runs/v3_HRL_Std_LoRA_E128_s42_20260620-121351'
d2 = 'C:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3/runs/v3_HRL_Std_NoLoRA_E128_s42_20260620-135743'

e1 = EventAccumulator(os.path.join(d1, os.listdir(d1)[0]))
e1.Reload()
e2 = EventAccumulator(os.path.join(d2, os.listdir(d2)[0]))
e2.Reload()

def get_last_n(e, tag, n=10):
    vals = [x.value for x in e.Scalars(tag)]
    return np.mean(vals[-n:]) if vals else 0.0

tags = ['Subtasks/Iron_Pct', 'Subtasks/Sword_Pct', 'Subtasks/Pickaxe_Pct', 'Episodes/Success_Rate']
print("=== Last 10 values ===")
for tag in tags:
    l_val = get_last_n(e1, tag)
    n_val = get_last_n(e2, tag)
    print(f"{tag}: LoRA={l_val:.3f}, NoLoRA={n_val:.3f}")
