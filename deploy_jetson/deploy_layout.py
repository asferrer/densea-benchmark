"""Folder convention of the deployment artifacts on the PC side.

Everything lives under ``RUNS_DIR/deploy_jetson``:

    models/        ONNX files (FP32 and Q/DQ) and the TensorRT engines copied back from the Jetson
    dd/            DiffusionDet EMA weights exported for the deployment
    results/       logs, trtexec outputs and prediction files copied back from the Jetson
    boot/<tag>/    ``<name>_val_preds.json`` files read by ``int8_bootstrap.py`` and ``int8_per_class.py``
    latency_5090/  raw per-model JSON of ``latency_5090.py``

``RUNS_DIR`` is configured through ``common.paths`` (mounted as ``/runs`` in the containers).
"""
from common import paths

DEPLOY_DIR = paths.RUNS_DIR / "deploy_jetson"
MODELS_DIR = DEPLOY_DIR / "models"
DD_DIR = DEPLOY_DIR / "dd"
RESULTS_DIR = DEPLOY_DIR / "results"
LATENCY_DIR = DEPLOY_DIR / "latency_5090"


def boot_dir(tag):
    """Folder with the VAL prediction files of one measurement set (for example ``15W``)."""
    return DEPLOY_DIR / "boot" / tag
