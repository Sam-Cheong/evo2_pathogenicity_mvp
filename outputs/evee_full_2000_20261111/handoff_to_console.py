"""Resume the existing EVEE experiment in the PowerShell running this script."""
import json
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
import psutil

output = Path(__file__).resolve().parent
repo = output.parents[1]
os.chdir(repo)
state = json.loads((output / "job_status.json").read_text(encoding="utf-8"))["state"]
if state in {"complete", "training"}:
    raise SystemExit(f"Job is {state}; leaving it alone.")

try:
    worker = psutil.Process(4288)
    launcher = psutil.Process(4280)
    if (abs(worker.create_time() - 1794467028.015554) > 0.01
            or abs(launcher.create_time() - 1794467027.8712032) > 0.01
            or worker.ppid() != launcher.pid
            or worker.cmdline()[1:] != ["-u", "scripts/run_evee_full.py", "run"]
            or Path(worker.cwd()).resolve() != repo):
        raise SystemExit("Worker identity changed. No process was terminated.")
    worker.terminate()
    worker.wait(timeout=10)
except psutil.NoSuchProcess:
    pass

with FileLock(output / ".full_run.lock", timeout=5):
    pass
print("Resuming EVEE in this console. Keep it open.", flush=True)
raise SystemExit(subprocess.call([sys.executable, "-u", "scripts/run_evee_full.py", "run"], cwd=repo))
