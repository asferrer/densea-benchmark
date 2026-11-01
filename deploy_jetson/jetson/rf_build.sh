#!/usr/bin/env bash
# TensorRT engine of RF-DETR (export_onnx_rfdetr.py, static shape 1x3x560x560) built with trtexec in the current mode.
# Usage: WORK_DIR=<deployment folder> bash rf_build.sh ONNX_NAME SUFFIX [trtexec options...]
#   e.g. bash rf_build.sh rfdetr_exp07 fp16 --fp16 ; bash rf_build.sh rfdetr_exp07 fp32
#        bash rf_build.sh rfdetr_exp07_qG int8 --int8 --fp16
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
N=$1; S=$2; shift; [ $# -gt 0 ] && shift
/usr/src/tensorrt/bin/trtexec --onnx=models/$N.onnx "$@" --saveEngine=models/${N}_$S.engine \
  --memPoolSize=workspace:2048 --skipInference > results/build_rf_${N}_$S.log 2>&1
echo "$(date -Is) ${N}_$S exit $?"; ls -la models/${N}_$S.engine 2>/dev/null
grep -iE "error|unsupported|overflow" results/build_rf_${N}_$S.log | sort | uniq -c | head -5
echo DONE
