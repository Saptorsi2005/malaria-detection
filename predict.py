import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # force CPU, skip GPU detection
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # suppress TF noise
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")  # disable slow graph compilation

"""
Malaria Detection — Prediction Module
======================================
Loads the saved best model and predicts on a single image.

Usage:
    python predict.py path/to/image.png
    python predict.py path/to/image.png --model models/custom_best.h5
"""

import sys
import argparse
import numpy as np
from pathlib import Path

from PIL import Image
# TensorFlow is imported lazily inside functions to keep Flask startup fast
# and to avoid aborting the server when tf is not yet needed.

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_malaria_model.h5"

# Label mapping (matches training: alphabetical folder order)
#   Parasitized → 0,  Uninfected → 1
CLASS_NAMES = ["Parasitized", "Uninfected"]
CLASS_DESCRIPTIONS = {
    "Parasitized": "⚠️  Malaria parasite detected in the blood sample.",
    "Uninfected": "✅ No malaria parasite detected. Sample appears healthy.",
}


def load_model(model_path: str) -> "tf.keras.Model":
    """Load a saved Keras model from disk."""
    import tensorflow as tf

    # Compatibility shim: TF 2.10+ saves DepthwiseConv2D with 'groups=1',
    # but older Keras versions don't recognise that argument.
    class _DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop("groups", None)   # strip unsupported arg silently
            super().__init__(**kwargs)

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run train.py first to generate the model."
        )
    print(f"[Model] Loading from: {path}")
    model = tf.keras.models.load_model(
        str(path),
        compile=False,
        custom_objects={"DepthwiseConv2D": _DepthwiseConv2DCompat},
    )
    print("[Model] Loaded successfully.\n")
    return model


def preprocess_image(image_path: str) -> "np.ndarray":
    """
    Load, resize to 224×224, convert to RGB, and normalise to [0, 1].
    Returns array of shape (1, 224, 224, 3) ready for inference.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc

    img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)     # shape: (1, 224, 224, 3)


def predict(model, img_array: "np.ndarray") -> dict:
    """
    Run inference and return a result dict:
    {
        "label":       "Parasitized" | "Uninfected",
        "class_index": 0 | 1,
        "confidence":  float (0–100),
        "raw_score":   float (sigmoid output, 0–1),
    }
    """
    raw_score: float = float(model.predict(img_array, verbose=0)[0][0])

    # Threshold at 0.5 — sigmoid output
    #   score → 0  ≈ Parasitized (class 0)
    #   score → 1  ≈ Uninfected  (class 1)
    class_index = 1 if raw_score >= 0.5 else 0
    label = CLASS_NAMES[class_index]
    confidence = raw_score * 100 if class_index == 1 else (1 - raw_score) * 100

    return {
        "label": label,
        "class_index": class_index,
        "confidence": round(confidence, 2),
        "raw_score": round(raw_score, 6),
    }


def print_result(result: dict, image_path: str) -> None:
    """Pretty-print the prediction result to the console."""
    print("=" * 55)
    print("  MALARIA DETECTION — PREDICTION RESULT")
    print("=" * 55)
    print(f"  Image       : {image_path}")
    print(f"  Prediction  : {result['label']}")
    print(f"  Confidence  : {result['confidence']:.2f}%")
    print(f"  Raw Score   : {result['raw_score']:.6f}")
    print("-" * 55)
    print(f"  {CLASS_DESCRIPTIONS[result['label']]}")
    print("=" * 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Malaria Detection — single-image prediction"
    )
    parser.add_argument(
        "image", type=str,
        help="Path to the blood smear image (PNG/JPG/BMP/TIFF)"
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL_PATH),
        help=f"Path to saved .h5 model (default: {DEFAULT_MODEL_PATH})"
    )
    return parser.parse_args()


# ─── Public API (importable by app.py) ───────────────────────────────────────

_cached_model = None

def get_cached_model(model_path: str = str(DEFAULT_MODEL_PATH)):
    """Return a cached model instance to avoid reloading on every request."""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model(model_path)
    return _cached_model


def predict_from_path(image_path: str, model_path: str = str(DEFAULT_MODEL_PATH)) -> dict:
    """
    High-level function used by app.py.
    Returns result dict with label, confidence, raw_score.
    """
    model = get_cached_model(model_path)
    img_array = preprocess_image(image_path)
    return predict(model, img_array)


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    loaded_model = load_model(args.model)
    img = preprocess_image(args.image)
    result = predict(loaded_model, img)
    print_result(result, args.image)
