#!/usr/bin/env bash
# Per-layer profile of TensorRT engines (why INT8 speeds up little): builds copies with --profilingVerbosity=detailed
# and exports the layer information (precision, tactic) and the time per layer. Analyze the output with prof_regions.py.
# Usage: WORK_DIR=<deployment folder> bash prof_layers.sh "engine_name|onnx_name|build flags|run inputs" ...
#   e.g. "rt_fp16|rtdetrv2_exp07_simplified|--fp16|" "rt_int8|rtdetrv2_exp07_int8mix_stem|--int8 --fp16|"
#        "rf_fp16|rfdetr_exp07|--fp16|--loadInputs=input:models/rf_in_input.bin"          (outputs in results/prof/)
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}" || exit 1
mkdir -p results/prof
T=/usr/src/tensorrt/bin/trtexec
sudo -n jetson_clocks; echo "$(date -Is) mode: $(sudo -n nvpmodel -q | head -1)"
for spec in "$@"; do
  IFS='|' read -r N ONNX FLAGS INPUTS <<< "$spec"
  E=models/prof_$N.engine; O=results/prof/$N
  $T --onnx=models/$ONNX.onnx $FLAGS --saveEngine=$E --memPoolSize=workspace:2048 --profilingVerbosity=detailed \
     --skipInference > $O.build.log 2>&1; echo "$(date -Is) build $N exit $?"
  $T --loadEngine=$E $INPUTS --warmUp=1000 --iterations=200 --useSpinWait --separateProfileRun --dumpProfile \
     --exportLayerInfo=$O.layers.json --exportProfile=$O.profile.json > $O.run.log 2>&1; echo "$(date -Is) run $N exit $?"
  grep -E "GPU Compute Time: min" $O.run.log | cut -c1-120
done
echo DONE
