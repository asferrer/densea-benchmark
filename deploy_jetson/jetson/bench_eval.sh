#!/usr/bin/env bash
# RT-DETRv2 on the Jetson: latency (trtexec), power (tegrastats) and VAL AP (eval_trt_val.py) of the FP16 and mixed
# INT8 engines in the current power mode, with jetson_clocks. The original clock configuration is stored first in
# results/jetson_clocks_orig.conf.
# Usage: WORK_DIR=<deployment folder> [PYTHON=<python with torch and pycocotools>] [MODEL=<base name>] \
#        bash bench_eval.sh MODE_LABEL [noap]        (outputs in results/<MODE_LABEL>_*)
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
B=${MODEL:-rtdetrv2_exp07}; S=${MODEL:+_$MODEL}   # log suffix only when MODEL is set
T=/usr/src/tensorrt/bin/trtexec; P=${PYTHON:-python3}; TAG=${1:-15W}
[ -f results/jetson_clocks_orig.conf ] || sudo -n jetson_clocks --store results/jetson_clocks_orig.conf
sudo -n jetson_clocks; sleep 5
echo "$(date -Is) mode: $(sudo -n nvpmodel -q 2>/dev/null | head -1)"; sudo -n jetson_clocks --show > results/${TAG}_clocks.txt 2>&1
sudo -n tegrastats --interval 100 --logfile results/${TAG}_tegra_idle.log & sleep 10; sudo -n tegrastats --stop; sleep 1
echo "idle: $(python3 power.py results/${TAG}_tegra_idle.log)"
for e in fp16 int8mix_stem; do
  sudo -n tegrastats --interval 100 --logfile results/${TAG}${S}_tegra_$e.log &
  $T --loadEngine=models/${B}_$e.engine --warmUp=2000 --duration=30 --useSpinWait \
     > results/${TAG}${S}_trtexec_$e.log 2>&1; echo "$(date -Is) trtexec $e exit $?"
  sudo -n tegrastats --stop; sleep 1
  grep -E "Throughput|GPU Compute Time" results/${TAG}${S}_trtexec_$e.log | tail -2 | cut -c1-150
  echo "power $e: $(python3 power.py results/${TAG}${S}_tegra_$e.log results/${TAG}${S}_trtexec_$e.log)"
done
[ "$2" = noap ] && { echo DONE; exit 0; }
for e in fp16 int8mix_stem; do
  PYTHONPATH=/usr/lib/python3.10/dist-packages $P eval_trt_val.py --engine models/${B}_$e.engine \
     --val data/val.json --data_root data --out results/${TAG}${S}_val_preds_$e.json > results/${TAG}${S}_eval_$e.log 2>&1
  echo "$(date -Is) eval $e exit $?"; tail -1 results/${TAG}${S}_eval_$e.log
done
echo DONE
