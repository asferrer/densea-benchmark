"""Input preprocessing and output decoding of the deployed RT-DETRv2 models (shared by quantization and by the
ONNX and TensorRT evaluations)."""
import cv2
import numpy as np

INPUT_H = 640
INPUT_W = 640


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 image -> contiguous float32 tensor (1, 3, 640, 640) in [0, 1].

    cv2 resize, ``INTER_AREA`` when the image is taller than the input and ``INTER_LINEAR`` otherwise.
    """
    interp = cv2.INTER_AREA if image_bgr.shape[0] > INPUT_H else cv2.INTER_LINEAR
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=interp)
    img = img.astype(np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])


SCORE_MIN = 0.001


def detections(image_id, labels, boxes, scores):
    """COCO detections (xywh) of one image for the model outputs with score above ``SCORE_MIN``."""
    keep = scores > SCORE_MIN
    return [{"image_id": image_id, "category_id": int(lab),
             "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "score": float(s)}
            for lab, b, s in zip(labels[keep], boxes[keep], scores[keep])]
