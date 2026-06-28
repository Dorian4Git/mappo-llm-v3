import sys
from tensorboard.backend.event_processing import event_accumulator
log_dir = sys.argv[1]
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()
tags = ['Wood', 'Stone', 'Pickaxe', 'Iron', 'Sword', 'Armor', 'Bridge', 'Enemy', 'Gold']
for t in tags:
    tag = f"Subtasks/{t}_Pct"
    if tag in ea.Tags()['scalars']:
        vals = [e.value for e in ea.Scalars(tag)]
        print(f"{t}: mean={sum(vals)/len(vals):.2f}, max={max(vals):.2f}")
