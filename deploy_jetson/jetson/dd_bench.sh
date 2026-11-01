#!/usr/bin/env bash
# DiffusionDet on the Jetson: latency and power (trtexec + tegrastats, shape 800x800 = 2,021 of the 2,185 VAL images)
# and VAL AP of the TensorRT engines and, with TORCH=1, of the PyTorch wrapper with the same noise.
# Layout under WORK_DIR: dd/ (eval_dd_val.py, export_onnx_diffusiondet.py, eval_utils.py, dump_inputs.py,
# densea_res50_jetson.yaml, code/ = DiffusionDet detectron2 project), models/, data/, venv_dd/, results/.
# Usage: WORK_DIR=<deployment folder> [NOAP=1] [TORCH=1 [WEIGHTS=<ema .pth>]] bash dd_bench.sh MODE_LABEL [engines...]
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
TAG=${1:-15W}; shift; ENGS=${@:-diffusiondet_exp07_fp16}
T=/usr/src/tensorrt/bin/trtexec; PY=$WORK_DIR/venv_dd/bin/python
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6 PYTHONPATH=/usr/lib/python3.10/dist-packages
export DD_CODE_DIR=$WORK_DIR/dd/code DD_CONFIG=$WORK_DIR/dd/densea_res50_jetson.yaml
VAL=../data/val.json; ROOT=../data
[ -f models/dd_in_image.bin ] || (cd dd && $PY dump_inputs.py --val $VAL --data_root $ROOT --out ../models)
sudo -n jetson_clocks; sleep 3; echo "$(date -Is) mode: $(sudo -n nvpmodel -q | head -1)"
for e in $ENGS; do
  sudo -n tegrastats --interval 100 --logfile results/${TAG}_tegra_$e.log &
  $T --loadEngine=models/$e.engine --shapes=image:1x3x800x800 --loadInputs=image:models/dd_in_image.bin,images_whwh:models/dd_in_images_whwh.bin,noise:models/dd_in_noise.bin --warmUp=2000 --duration=30 --useSpinWait \
     > results/${TAG}_trtexec_$e.log 2>&1; echo "$(date -Is) trtexec $e exit $?"
  sudo -n tegrastats --stop; sleep 1
  grep -E "GPU Compute Time" results/${TAG}_trtexec_$e.log | tail -1 | cut -c1-150
  echo "power $e: $(python3 power.py results/${TAG}_tegra_$e.log results/${TAG}_trtexec_$e.log)"
done
[ "$NOAP" = 1 ] && { echo DONE; exit 0; }
for e in $ENGS; do
  (cd dd && $PY eval_dd_val.py --backend trt --engine ../models/$e.engine --val $VAL --data_root $ROOT \
     --out ../results/${TAG}_dd_val_$e.json) > results/${TAG}_dd_eval_$e.log 2>&1
  echo "$(date -Is) eval $e exit $?"; tail -1 results/${TAG}_dd_eval_$e.log | cut -c1-300
done
if [ "$TORCH" = 1 ]; then
  (cd dd && $PY eval_dd_val.py --backend torch --weights "${WEIGHTS:?set WEIGHTS to the EMA weights}" --val $VAL \
     --data_root $ROOT --out ../results/${TAG}_dd_val_torch.json) > results/${TAG}_dd_eval_torch.log 2>&1
  echo "$(date -Is) eval torch exit $?"; tail -1 results/${TAG}_dd_eval_torch.log | cut -c1-300
fi
echo DONE
