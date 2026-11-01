"""GPU monitor: log the GPU state every N seconds and detect risky conditions.

Writes one CSV row per sample into ``<log-dir>/gpu_monitor.csv``. If the GPU stops responding ("GPU is lost") or
temperature or power stay above their limits for three consecutive samples, the monitor appends a line to
``<log-dir>/ALERT.txt`` and, when ``--kill-pid`` is given, terminates that process tree to protect the card.

    python run/gpu_guard/gpu_monitor.py --log-dir runs/gpu_logs --max-temp 83 --max-power 440 [--kill-pid PID]

Every training of the benchmark is meant to run under this monitor (``run/run_phase.py`` starts it).
"""
import argparse
import csv
import datetime as dt
import os
import signal
import subprocess
import time

FIELDS = ("timestamp,power.draw,power.limit,temperature.gpu,temperature.memory,"
          "clocks.sm,clocks.mem,utilization.gpu,memory.used,pcie.link.gen.current,"
          "pcie.link.width.current,clocks_event_reasons.active")


def sample():
    """Return ``(row, "")`` from nvidia-smi or ``(None, reason)`` when the GPU does not respond."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return None, f"nvidia-smi unavailable: {exc}"
    text = (out.stdout + out.stderr).strip()
    if out.returncode != 0 or "lost" in text.lower() or "No devices" in text:
        return None, text
    return [c.strip() for c in text.splitlines()[0].split(",")], ""


def terminate(pid):
    """Terminate a process tree on Windows (taskkill) or a process group on POSIX."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def alert(log_dir, reason, kill_pid):
    """Record the alert and, if requested, stop the monitored process."""
    with open(os.path.join(log_dir, "ALERT.txt"), "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now().isoformat()} {reason}\n")
    if kill_pid:
        terminate(kill_pid)


def risk(row, max_temp, max_power):
    """Reason why the sample is out of limits, or an empty string."""
    try:
        temp, power = float(row[3]), float(row[1])
    except ValueError:
        return ""
    if temp >= max_temp:
        return f"temperature {temp} >= {max_temp}"
    if power >= max_power:
        return f"power {power} >= {max_power}"
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    ap.add_argument("--max-temp", type=float, default=83.0, help="GPU temperature limit (C)")
    ap.add_argument("--max-power", type=float, default=440.0, help="power draw limit (W)")
    ap.add_argument("--kill-pid", type=int, default=0, help="process to terminate on alert")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until stopped")
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True)
    path = os.path.join(a.log_dir, "gpu_monitor.csv")
    is_new = not os.path.exists(path)
    t0, strikes = time.time(), 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(FIELDS.split(","))
        while not a.duration or time.time() - t0 < a.duration:
            row, err = sample()
            if row is None:
                alert(a.log_dir, f"GPU not responding: {err}", a.kill_pid)
                break
            writer.writerow(row)
            f.flush()
            reason = risk(row, a.max_temp, a.max_power)
            strikes = strikes + 1 if reason else 0
            if strikes >= 3:  # three consecutive samples out of limits
                alert(a.log_dir, reason, a.kill_pid)
                strikes = 0
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
