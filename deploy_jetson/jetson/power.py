"""Mean power from a tegrastats log and energy per inference.

Rails: VDD_IN (the whole board), VDD_CPU_GPU_CV and VDD_SOC. With a trtexec log it also reports the throughput and
the board energy per inference, VDD_IN / throughput.

Usage: python3 power.py tegrastats.log [trtexec.log]
"""
import re
import sys

RAILS = ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")


def rail_means(lines):
    """Mean power in W of each rail present in the tegrastats lines, and the number of VDD_IN samples."""
    values = {r: [] for r in RAILS}
    for line in lines:
        for rail in RAILS:
            m = re.search(rail + r" (\d+)mW/", line)
            if m:
                values[rail].append(int(m.group(1)))
    out = {r: round(sum(v) / len(v) / 1000, 2) for r, v in values.items() if v}
    out["samples"] = len(values["VDD_IN"])
    return out


def throughput_qps(trtexec_log):
    """Throughput in queries per second reported by trtexec."""
    return float(re.search(r"Throughput: ([\d.]+) qps", trtexec_log).group(1))


def summarize(tegrastats_lines, trtexec_log=None):
    out = rail_means(tegrastats_lines)
    if trtexec_log is not None:
        out["qps"] = throughput_qps(trtexec_log)
        out["mJ_per_inference_board"] = round(1000 * out["VDD_IN"] / out["qps"], 1)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    log = None
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            log = f.read()
    print(summarize(lines, log))


if __name__ == "__main__":
    main()
