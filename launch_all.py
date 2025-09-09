# launch_all.py
import os, sys, yaml, subprocess, time
from pathlib import Path

CFG_PATH = Path("config") / "accounts.yaml"
LOG_DIR = Path("logs")

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def spawn(name, entry):
    """
    Spawns shim_run.py which in turn runs the target script.
    We don't set any MT5_* env vars here (keeping the scripts' own credentials/paths).
    If you ever want to override per-process later, you can set them here.
    """
    env = os.environ.copy()
    env["TARGET_SCRIPT"] = entry["script_path"]

    LOG_DIR.mkdir(exist_ok=True)
    out = open(LOG_DIR / f"{name}.out.log", "a", buffering=1)
    err = open(LOG_DIR / f"{name}.err.log", "a", buffering=1)

    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        [sys.executable, "shim_run.py"],
        env=env, stdout=out, stderr=err,
        creationflags=flags
    )

def main():
    cfg = load_cfg(CFG_PATH)
    algos = cfg.get("algos", {})
    if not algos:
        print("No algos found in config/accounts.yaml.")
        sys.exit(1)

    procs = {name: spawn(name, entry) for name, entry in algos.items()}
    print("Launched:", ", ".join(procs.keys()))

    try:
        while True:
            time.sleep(5)
            for name, p in list(procs.items()):
                if p.poll() is not None:
                    code = p.returncode
                    print(f"{name} exited ({code}); restarting in 3s…")
                    time.sleep(3)
                    procs[name] = spawn(name, algos[name])
    except KeyboardInterrupt:
        print("\nStopping…")
        for p in procs.values():
            try:
                p.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    main()
