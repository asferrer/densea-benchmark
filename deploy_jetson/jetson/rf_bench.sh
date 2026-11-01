#!/usr/bin/env bash
# RF-DETR on the Jetson: latency and power (trtexec with a real input + tegrastats) and VAL AP of the engines.
# Usage: WORK_DIR=<deployment folder> [PYTHON=<python with torch and pycocotools>] [NOAP=1] \
#        bash rf_bench.sh MODE_LABEL engine1 [engine2...]   (NOAP=1: latency and power only; outputs in results/<MODE_LABEL>_*)
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
TAG=${1:-15W}; shift; ENGS=${@:-rfdetr_exp07_fp16}
T=/usr/src/tensorrt/bin/trtexec; P=${PYTHON:-python3}
export PYTHONPATH=/usr/lib/python3.10/dist-packages
[ -f models/rf_in_input.bin ] || $P -c "
import json, numpy as np; from PIL import Image; from eval_rfdetr_val import prepare
v = json.load(open('data/val.json')); im = min(v['images'], key=lambda d: d['id'])
prepare(Image.open('data/' + im['file_name']).convert('RGB')).tofile('models/rf_in_input.bin')"
sudo -n jetson_clocks; sleep 3; echo "$(date -Is) mode: $(sudo -n nvpmodel -q | head -1)"
for e in $ENGS; do
  sudo -n tegrastats --interval 100 --logfile results/${TAG}_tegra_$e.log &
  $T --loadEngine=models/$e.engine --loadInputs=input:models/rf_in_input.bin --warmUp=2000 --duration=30 --useSpinWait \
     > results/${TAG}_trtexec_$e.log 2>&1; echo "$(date -Is) trtexec $e exit $?"
  sudo -n tegrastats --stop; sleep 1
  grep -E "GPU Compute Time" results/${TAG}_trtexec_$e.log | tail -1 | cut -c1-150
  echo "power $e: $(python3 power.py results/${TAG}_tegra_$e.log results/${TAG}_trtexec_$e.log)"
done
[ "$NOAP" = 1 ] && { echo DONE; exit 0; }
for e in $ENGS; do
  $P eval_rfdetr_val.py --engine models/$e.engine --val data/val.json --data_root data \
     --out results/${TAG}_rf_val_$e.json > results/${TAG}_rf_eval_$e.log 2>&1
  echo "$(date -Is) eval $e exit $?"; tail -1 results/${TAG}_rf_eval_$e.log | cut -c1-300
done
echo DONE
