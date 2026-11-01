#!/usr/bin/env bash
# TensorRT engine of DiffusionDet (wrapper of export_onnx_diffusiondet.py) with a dynamic profile covering the four
# VAL shapes (800x800 in 2,021 of 2,185 images; 544/640/736x960).
# Usage: WORK_DIR=<deployment folder> bash build_dd_engine.sh fp16|fp32 [ONNX_NAME]   (default diffusiondet_exp07)
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
PREC=${1:-fp16}; N=${2:-diffusiondet_exp07}
S=""; [ "$N" = diffusiondet_exp07 ] || S="_${N#diffusiondet_}"   # log suffix except for exp07
FLAG=""; [ "$PREC" = fp16 ] && FLAG="--fp16"
/usr/src/tensorrt/bin/trtexec --onnx=models/$N.onnx $FLAG --saveEngine=models/${N}_$PREC.engine \
  --minShapes=image:1x3x544x800 --optShapes=image:1x3x800x800 --maxShapes=image:1x3x800x960 \
  --memPoolSize=workspace:2048 --skipInference > results/build_dd${S}_$PREC.log 2>&1
echo "$(date -Is) $PREC exit $?"; ls -la models/${N}_$PREC.engine 2>/dev/null
grep -iE "error|unsupported|plugin" results/build_dd${S}_$PREC.log | head -5
echo DONE
