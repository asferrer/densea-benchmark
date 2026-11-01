#!/usr/bin/env bash
# Mixed-precision engine of DiffusionDet: backbone and FPN in FP16, the nodes that match PATTERN (default /head/*,
# the six heads) in FP32 through --layerPrecisions. Only --layerPrecisions is used: with --layerOutputTypes the Cast
# layers fail. CONSTR is the precision-constraints mode (default obey); the engine of the paper uses CONSTR=prefer,
# because with obey the RoiAlign plugin offers no FP32 format inside the constraint.
# Usage: WORK_DIR=<deployment folder> [CONSTR=prefer] bash build_dd_mixed.sh SUFFIX "PATTERN[,PATTERN]" [ONNX_NAME]
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
SUF=$1; PAT=${2:-/head/*}; N=${3:-diffusiondet_exp07}
S=""; case "$N" in diffusiondet_exp07*) ;; *) S="_${N#diffusiondet_}";; esac
LP=$(echo "$PAT" | tr "," "\n" | sed "s/$/:fp32/" | paste -sd,)
/usr/src/tensorrt/bin/trtexec --onnx=models/$N.onnx --fp16 --precisionConstraints=${CONSTR:-obey} \
  --layerPrecisions="$LP" --saveEngine=models/${N}_$SUF.engine \
  --minShapes=image:1x3x544x800 --optShapes=image:1x3x800x800 --maxShapes=image:1x3x800x960 \
  --memPoolSize=workspace:2048 --skipInference > results/build_dd${S}_$SUF.log 2>&1
echo "$(date -Is) $SUF exit $?"; ls -la models/${N}_$SUF.engine 2>/dev/null; grep -iE "error" results/build_dd${S}_$SUF.log | head -3
echo DONE
