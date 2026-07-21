import os, glob
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('runs/v3_HRL_Std_LoRA_E128_s42_20260720-190702', size_guidance={'scalars': 0})
ea.Reload()
tags = ea.Tags().get('scalars', [])
print(tags)
