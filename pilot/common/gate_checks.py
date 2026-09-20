"""Environment gates run at the start of every job on the cluster.

Every failure mode here has actually happened on this cluster: jobs COMPLETED
while writing nothing (missing mount), jobs killed by a full disk, jobs
launched onto nodes without a visible GPU. Gates fail LOUDLY and early.
"""
import json
import os
import socket
import subprocess
import sys
import time
import uuid


def check_write_readback(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    token = uuid.uuid4().hex
    path = os.path.join(out_dir, f".gate_sentinel_{token}")
    with open(path, "w") as f:
        f.write(token)
        f.flush()
        os.fsync(f.fileno())
    with open(path) as f:
        back = f.read()
    os.remove(path)
    if back != token:
        raise RuntimeError(f"GATE FAIL: write-readback mismatch in {out_dir}")


def check_disk(path: str, min_gb: float) -> float:
    st = os.statvfs(path)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    if free_gb < min_gb:
        raise RuntimeError(
            f"GATE FAIL: only {free_gb:.1f} GB free at {path}, need {min_gb} GB. "
            "A full disk kills running jobs on this cluster - aborting before contributing."
        )
    return free_gb


def check_gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError("GATE FAIL: nvidia-smi not found")
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"GATE FAIL: no GPU visible: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def run_all_gates(out_dir: str, min_free_gb: float = 40.0, need_gpu: bool = True) -> dict:
    info = {
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    check_write_readback(out_dir)
    info["free_gb_at_start"] = round(check_disk(out_dir, min_free_gb), 1)
    if need_gpu:
        info["gpu"] = check_gpu()
    with open(os.path.join(out_dir, "gate_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[gates] OK host={info['host']} free={info['free_gb_at_start']}GB "
          f"gpu={info.get('gpu', 'n/a')}", flush=True)
    return info
