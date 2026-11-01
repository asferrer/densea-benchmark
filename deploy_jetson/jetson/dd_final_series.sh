#!/usr/bin/env bash
# Final DiffusionDet series on the Jetson (the model of the paper, exp06v11_synth100_real100_s3, ONNX base name
# diffusiondet_exp06v11s3). All of VAL.
# Part a (no reboot): FP32, FP16 and mixed engines (/head/* in FP32, CONSTR=prefer); 15 W with AP (and the PyTorch
# wrapper with the same noise), then MAXN_SUPER without AP.
# Part b, after switching to 7 W (`echo YES | sudo nvpmodel -m 3`, which reboots): latency and power at 7 W.
# Afterwards switch back to 15 W (`echo YES | sudo nvpmodel -m 0`, another reboot).
# Usage: WORK_DIR=<deployment folder> bash dd_final_series.sh a|b
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
M=diffusiondet_exp06v11s3; ENGS="${M}_fp32 ${M}_mixhead ${M}_fp16"
if [ "$1" = a ]; then
  bash build_dd_engine.sh fp32 $M; bash build_dd_engine.sh fp16 $M
  CONSTR=prefer bash build_dd_mixed.sh mixhead "/head/*" $M
  for e in $ENGS; do [ -s models/$e.engine ] || { echo "missing models/$e.engine"; echo DONE_A; exit 1; }; done
  TORCH=1 WEIGHTS=$WORK_DIR/models/${M}_ema.pth bash dd_bench.sh 15W_final $ENGS
  echo NO | sudo -n nvpmodel -m 2 2>&1 | tail -1; echo "$(date -Is) $(sudo -n nvpmodel -q | head -1)"
  NOAP=1 bash dd_bench.sh MAXN_SUPER_final $ENGS
  echo NO | sudo -n nvpmodel -m 0 2>&1 | tail -1
  sudo -n jetson_clocks --restore results/jetson_clocks_orig.conf
  echo "$(date -Is) part a done: $(sudo -n nvpmodel -q | head -1)"; echo DONE_A
elif [ "$1" = b ]; then
  echo "$(date -Is) $(sudo -n nvpmodel -q | head -1)"
  NOAP=1 bash dd_bench.sh 7W_final $ENGS
  sudo -n jetson_clocks --restore results/jetson_clocks_orig.conf
  echo DONE_B
fi
