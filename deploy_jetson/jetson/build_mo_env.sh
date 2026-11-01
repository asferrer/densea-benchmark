#!/usr/bin/env bash
# Separate virtualenv with ModelOpt 0.19 (the version used on the PC) to quantize with Q/DQ nodes on the CPU of the
# Jetson itself (quant_rf.py), without touching the base environment. The system Python has no ensurepip, so the
# virtualenv is created from the base Python.
# Usage: WORK_DIR=<deployment folder> BASE_PYTHON=<python with venv support> bash build_mo_env.sh   (creates venv_mo)
set -e
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}"
V=$PWD/venv_mo
rm -rf $V; ${BASE_PYTHON:?set BASE_PYTHON to a Python with venv support} -m venv $V
$V/bin/pip install -q --upgrade "pip<25" "setuptools<70" wheel
$V/bin/pip install -q nvidia-modelopt==0.19.0 --no-deps
$V/bin/pip install -q "numpy<2" "scipy<1.15" onnx==1.17.0 onnxruntime==1.20.1 onnx-graphsurgeon onnxmltools==1.12.0 \
  onnxconverter-common pydantic pulp ml_dtypes cppimport ninja regex rich safetensors cloudpickle tqdm packaging pillow
$V/bin/python -c "from modelopt.onnx.quantization import quantize; import modelopt, onnx, onnxruntime as ort, numpy; print('OK', modelopt.__version__, onnx.__version__, ort.__version__, numpy.__version__)"
