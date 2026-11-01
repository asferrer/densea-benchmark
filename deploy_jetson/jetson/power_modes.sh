#!/usr/bin/env bash
# Latency and power (no AP) of the RT-DETRv2 engines at 15 W and MAXN_SUPER, then restores the Jetson to 15 W and the
# original clocks (results/jetson_clocks_orig.conf). The 7 W mode asks for a reboot: switch to it with
# `echo YES | sudo nvpmodel -m 3`, wait for the reboot, check `nvpmodel -q`, run `bash bench_eval.sh 7W noap`, and go
# back with `echo YES | sudo nvpmodel -m 0` (another reboot).
# Usage: WORK_DIR=<deployment folder> [MODEL=<base name>] bash power_modes.sh
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
for spec in "0 15W" "2 MAXN_SUPER"; do
  set -- $spec
  echo NO | sudo -n nvpmodel -m $1 2>&1 | tail -2   # NO: do not reboot (modes 0 and 2 do not need it)
  bash bench_eval.sh $2 noap
done
echo NO | sudo -n nvpmodel -m 0 2>&1 | tail -2
sudo -n jetson_clocks --restore results/jetson_clocks_orig.conf
echo "$(date -Is) restored: $(sudo -n nvpmodel -q | head -1)"
echo DONE
