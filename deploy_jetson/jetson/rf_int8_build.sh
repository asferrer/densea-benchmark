#!/usr/bin/env bash
# Explicit INT8 (Q/DQ) variants of RF-DETR, quantized with ModelOpt 0.19 on the CPU of the Jetson and calibrated on
# 100 TRAIN images (calib/train_100 = images of deploy_jetson/calibration/calib_100.json copied with
# `make_calibration_set.py --copy-to`). Only MatMul / Gemm / Conv are quantized and the attention matmuls (Q.K, A.V)
# stay in FP16 (--disable_mha_qdq). The variants differ in which linear layers are quantized:
#   F  attention projections (query, key, value, output.dense)
#   G  F + mlp/fc1
#   H  F + the whole MLP
# Each one is built with trtexec (--int8 --fp16) and benchmarked at the current power mode with rf_bench.sh.
# Usage: WORK_DIR=<deployment folder> [PYTHON=<python with torch and pycocotools>] bash rf_int8_build.sh
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
P=venv_mo/bin/python; C="--calib calib/train_100 --n 100 --method entropy --op_types MatMul,Gemm,Conv --disable_mha_qdq"
AT='.*/attention/(attention/(query|key|value)|output/dense).*'
for spec in "F|$AT" "G|$AT,.*/mlp/fc1/.*" "H|$AT,.*/mlp/.*"; do
  IFS='|' read -r V R <<< "$spec"
  $P quant_rf.py --onnx models/rfdetr_exp07.onnx --out models/rfdetr_exp07_q$V.onnx $C --only "$R" > results/quant_q$V.log 2>&1 \
    || { echo "q$V FAILED"; tail -2 results/quant_q$V.log; continue; }
  echo "q$V: $(grep -m1 'Quantized type counts' results/quant_q$V.log | cut -c1-100)"
  bash rf_build.sh rfdetr_exp07_q$V int8 --int8 --fp16 | head -1
done
bash rf_bench.sh 15W rfdetr_exp07_qF_int8 rfdetr_exp07_qG_int8 rfdetr_exp07_qH_int8
