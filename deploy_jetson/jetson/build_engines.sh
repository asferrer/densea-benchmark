#!/usr/bin/env bash
# TensorRT engines of RT-DETRv2 built with trtexec in the current power mode.
# FP16 from the FP32 ONNX; mixed INT8 from the ONNX with ModelOpt Q/DQ nodes (backbone/conv1 kept out of INT8).
# Usage: WORK_DIR=<deployment folder> [MODEL=<base name>] bash build_engines.sh
#   expects models/<MODEL>_simplified.onnx and models/<MODEL>_int8mix_stem.onnx; MODEL defaults to rtdetrv2_exp07.
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
B=${MODEL:-rtdetrv2_exp07}
T=/usr/src/tensorrt/bin/trtexec
echo "$(date -Is) mode: $(sudo -n nvpmodel -q 2>/dev/null | head -1)"
$T --onnx=models/${B}_simplified.onnx --fp16 --saveEngine=models/${B}_fp16.engine \
   --memPoolSize=workspace:2048 --skipInference > results/build_${B}_fp16.log 2>&1; echo "$(date -Is) fp16 exit $?"
$T --onnx=models/${B}_int8mix_stem.onnx --int8 --fp16 --saveEngine=models/${B}_int8mix_stem.engine \
   --memPoolSize=workspace:2048 --skipInference > results/build_${B}_int8mix.log 2>&1; echo "$(date -Is) int8mix exit $?"
ls -la models/*.engine
echo DONE
