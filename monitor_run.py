import os
import sys
import time
import subprocess
import shutil
import psutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ASSETS_DIR = r"C:\Users\doria\OneDrive\Documents\school\master IS\Semestre 4\Travail de Master\rev_v3\meet_20260625\assets"
PLOTS_DIR = os.path.join(ASSETS_DIR, "plots")

# Target step: 550 * 128 * 256 = 18022400
TARGET_STEP = 18022400
PROJECT_DIR = "C:/PROJECTS/_SCHOOL/MasterIS/TM/mappo-llm-v3"
RUNS_BASE_DIR = os.path.join(PROJECT_DIR, "runs")

def get_current_step():
    all_runs = [os.path.join(RUNS_BASE_DIR, d) for d in os.listdir(RUNS_BASE_DIR) if "v3_HRL_Std_LoRA" in d]
    run_dir = max(all_runs, key=os.path.getmtime) if all_runs else None
    
    if run_dir is None or not os.path.exists(run_dir):
        return 0
    event_files = [f for f in os.listdir(run_dir) if "tfevents" in f]
    if not event_files:
        return 0
    event_file = os.path.join(run_dir, event_files[0])
    e = EventAccumulator(event_file)
    e.Reload()
    try:
        events = e.Scalars('Rewards/Avg_Env_Reward')
        return events[-1].step
    except:
        return 0

def kill_run_hrl():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info.get('cmdline')
            if cmd and 'python' in proc.info['name'].lower() and 'scripts/run_hrl.py' in cmd:
                print(f"Killing process {proc.info['pid']}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def clean_assets():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    for f in os.listdir(ASSETS_DIR):
        if (f.startswith("hrl") or f.startswith("Subtask")) and f.endswith(".png"):
            os.remove(os.path.join(ASSETS_DIR, f))

def run_plotting():
    subprocess.run(["C:/Users/doria/miniconda3/envs/tm/python.exe", "plot_comparison.py"], cwd=PROJECT_DIR)

print("Starting background monitor...")
while True:
    step = get_current_step()
    print(f"Current step: {step} / {TARGET_STEP}")
    if step >= TARGET_STEP:
        print("Target reached. Terminating process...")
        kill_run_hrl()
        
        print("Cleaning old assets...")
        clean_assets()
        
        print("Running plot_comparison.py...")
        run_plotting()
        print("Done monitoring.")
        sys.exit(0)
    
    time.sleep(300) # Check every 5 minutes
