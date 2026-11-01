#!/usr/bin/env bash
# Final RT-DETRv2 series on the Jetson (the model of the paper, exp08_real_synth_minor_s3, ONNX base name
# rtdetrv2_exp08s3). All of VAL.
# Part a (no reboot): 15 W with AP, then MAXN_SUPER without AP (the AP does not depend on the power mode).
# Part b, after switching to 7 W (`echo YES | sudo nvpmodel -m 3`, which reboots): latency and power at 7 W.
# Afterwards switch back to 15 W (`echo YES | sudo nvpmodel -m 0`, another reboot).
# Requires the engines of build_engines.sh. Usage: WORK_DIR=<deployment folder> [MODEL=<base name>] bash rt_final_series.sh a|b
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
export MODEL=${MODEL:-rtdetrv2_exp08s3}
if [ "$1" = a ]; then
  bash bench_eval.sh 15W_final
  echo NO | sudo -n nvpmodel -m 2 2>&1 | tail -1; echo "$(date -Is) $(sudo -n nvpmodel -q | head -1)"
  bash bench_eval.sh MAXN_SUPER_final noap
  echo NO | sudo -n nvpmodel -m 0 2>&1 | tail -1
  sudo -n jetson_clocks --restore results/jetson_clocks_orig.conf
  echo "$(date -Is) part a done: $(sudo -n nvpmodel -q | head -1)"; echo DONE_A
elif [ "$1" = b ]; then
  echo "$(date -Is) $(sudo -n nvpmodel -q | head -1)"
  bash bench_eval.sh 7W_final noap
  sudo -n jetson_clocks --restore results/jetson_clocks_orig.conf
  echo DONE_B
fi
