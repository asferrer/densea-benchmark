#!/usr/bin/env bash
# PyTorch environment of DiffusionDet on the Jetson: a virtualenv with --system-site-packages on top of the base
# Python environment (which is not modified), torchvision 0.20.1 built with CUDA for sm_87 (the prebuilt one has no
# roi_align on CUDA) and detectron2 at the commit pinned in the DiffusionDet training container.
# Usage: WORK_DIR=<deployment folder> BASE_PYTHON=<python with torch and CUDA> bash build_dd_env.sh
set -e
cd "${WORK_DIR:?set WORK_DIR to the deployment folder}"
BASE=${BASE_PYTHON:?set BASE_PYTHON to a Python with torch and CUDA}
[ -d venv_dd ] || $BASE -m venv --system-site-packages venv_dd
P=$WORK_DIR/venv_dd/bin/python
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.7" MAX_JOBS=3 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
echo "$(date -Is) torchvision"
[ -d src/vision ] || git clone -q --depth 1 --branch v0.20.1 https://github.com/pytorch/vision src/vision
(cd src/vision && BUILD_VERSION=0.20.1 $P -m pip install -q --no-build-isolation --no-deps . > ../../results/build_torchvision.log 2>&1)
$P -c "import torch, torchvision; from torchvision.ops import roi_align; x=torch.rand(1,4,16,16,device=\"cuda\"); b=torch.tensor([[0,0,0,8,8.]],device=\"cuda\"); print(\"torchvision\", torchvision.__version__, roi_align(x,b,(2,2)).shape)"
echo "$(date -Is) detectron2"
[ -d src/detectron2 ] || (git clone -q https://github.com/facebookresearch/detectron2 src/detectron2 && cd src/detectron2 && git checkout -q 8a9d885b3d4dcf1bef015f0593b872ed8d32b4ab)
$P -m pip install -q "fvcore<0.1.6" iopath omegaconf hydra-core yacs termcolor black > results/pip_dd.log 2>&1 || true
(cd src/detectron2 && $P -m pip install -q --no-build-isolation --no-deps -e . > ../../results/build_detectron2.log 2>&1)
$P -c "import detectron2, torchvision; from detectron2 import _C; print(\"detectron2\", detectron2.__version__, \"ok\")"
echo "$(date -Is) DONE"
