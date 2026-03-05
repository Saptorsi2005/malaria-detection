"""
Malaria Detection using Deep Learning — Training Script
=======================================================
Trains two models:
  - Model A: Custom CNN
  - Model B: Transfer Learning with MobileNetV2
Automatically selects the best model and saves it as 'models/best_malaria_model.h5'.

Dataset structure expected:
    dataset/
    ├── Parasitized/
    └── Uninfected/

Usage:
    python train.py
"""

import os
import sys
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Tuple, Dict

import tensorflow as tf
from tensorflow import keras
# Use tf.keras throughout — Keras 3 dropped keras.preprocessing entirely
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Dense, GlobalAveragePooling2D, Flatten, Input,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
)
from sklearn.model_selection import train_test_split
from PIL import Image

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Global Configuration ─────────────────────────────────────────────────────
IMG_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 8           # reduced for 4GB VRAM (RTX 2050)
EPOCHS_CNN: int = 40
EPOCHS_TRANSFER: int = 30
FINE_TUNE_EPOCHS: int = 15
LEARNING_RATE: float = 1e-3
FINE_TUNE_LR: float = 1e-5

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Label mapping — matches Keras flow_from_directory alphabetical order:
#   Parasitized → 0,  Uninfected → 1
CLASS_NAMES = ["Parasitized", "Uninfected"]

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & EXPLORATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_dataset() -> None:
    """Check that the dataset directory has the expected structure."""
    for cls in CLASS_NAMES:
        cls_dir = DATASET_DIR / cls
        if not cls_dir.exists():
            sys.exit(
                f"[ERROR] Missing directory: {cls_dir}\n"
                "Please place your images as described in the README."
            )
        count = len(list(cls_dir.glob("*")))
        if count == 0:
            sys.exit(f"[ERROR] No images found in {cls_dir}")
        print(f"  ✓ {cls}: {count} images")


def collect_image_paths() -> Tuple[list, list]:
    """Return (file_paths, labels) for all images in the dataset folder."""
    paths, labels = [], []
    label_map = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    for cls in CLASS_NAMES:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
            for img_path in (DATASET_DIR / cls).glob(ext):
                paths.append(str(img_path))
                labels.append(label_map[cls])
    return paths, labels


def summarize_image_dimensions(paths: list, sample_size: int = 50) -> None:
    """Print min/max/mean image dimensions from a random sample."""
    sample = random.sample(paths, min(sample_size, len(paths)))
    widths, heights = [], []
    for p in sample:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass
    print(
        f"\n[Image Dimensions (sample={len(sample)})]"
        f"\n  Width  — min:{min(widths)}, max:{max(widths)}, mean:{np.mean(widths):.1f}"
        f"\n  Height — min:{min(heights)}, max:{max(heights)}, mean:{np.mean(heights):.1f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(labels: list) -> None:
    """Bar chart of sample counts per class."""
    counts = [labels.count(i) for i in range(len(CLASS_NAMES))]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=["#e74c3c", "#2ecc71"], edgecolor="black", width=0.5)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Image Count")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(cnt), ha="center")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)
    print("[Saved] class_distribution.png")


def plot_sample_images(paths: list, labels: list, n: int = 8) -> None:
    """Grid of n random sample images, labelled by class."""
    indices = random.sample(range(len(paths)), min(n, len(paths)))
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6))
    axes = axes.flatten()
    for ax, idx in zip(axes, indices):
        img = Image.open(paths[idx]).resize(IMG_SIZE)
        ax.imshow(img)
        ax.set_title(CLASS_NAMES[labels[idx]], fontsize=9,
                     color="#e74c3c" if labels[idx] == 0 else "#2ecc71")
        ax.axis("off")
    fig.suptitle("Sample Images from Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "sample_images.png", dpi=120)
    plt.close(fig)
    print("[Saved] sample_images.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def build_generators(
    train_paths: list,
    val_paths: list,
    train_labels: list,
    val_labels: list,
) -> Tuple[any, any]:
    """
    Create Keras ImageDataGenerators.
    • Training: augmentation + normalization
    • Validation: normalization only
    """
    # Augmentation — only for training (prevents overfitting on tiny dataset)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.1,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Flow from Keras flow_from_directory, using the dataset root
    # We use flow_from_directory on the dataset dir (binary mode)
    train_gen = train_datagen.flow_from_directory(
        directory=str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset=None,            # we split manually below
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        directory=str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset=None,
        shuffle=False,
    )
    return train_gen, val_gen


def build_generators_split() -> Tuple[any, any, int, int]:
    """
    Build train/val generators using validation_split on ImageDataGenerator.
    Returns (train_gen, val_gen, n_train_steps, n_val_steps).
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.1,
        fill_mode="nearest",
        validation_split=0.2,
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        directory=str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        directory=str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    n_train = train_gen.samples
    n_val = val_gen.samples
    print(f"\n[Data Split] Train: {n_train} | Validation: {n_val}")
    return train_gen, val_gen, n_train, n_val


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

def build_cnn_model() -> Sequential:
    """
    Model A — Custom CNN
    Architecture: 3× (Conv2D → BN → MaxPool) → Dropout → Dense → Sigmoid
    Optimised for a small dataset with heavy regularisation.
    """
    model = Sequential(name="Custom_CNN")

    # Block 1
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                     input_shape=(*IMG_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Classification head
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))   # binary output

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_mobilenetv2_model() -> Model:
    """
    Model B — Transfer Learning with MobileNetV2 (ImageNet weights).
    Phase 1: freeze base, train only the classification head.
    Phase 2: unfreeze top layers for fine-tuning.
    Returns the model with only Phase-1 layers trainable.
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False    # freeze all base layers initially

    # Custom classification head
    inputs = Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="MobileNetV2_Transfer")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def unfreeze_top_layers(model: Model, n_layers: int = 30) -> Model:
    """Unfreeze the top n_layers of the MobileNetV2 base for fine-tuning."""
    base = model.layers[1]          # MobileNetV2 is layer index 1
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    print(f"\n[Fine-Tune] Unfroze top {n_layers} layers of MobileNetV2 base.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str) -> list:
    """Return standard training callbacks for both models."""
    checkpoint_path = str(MODELS_DIR / f"{model_name}_best.h5")
    return [
        EarlyStopping(
            monitor="val_auc",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: keras.Model,
    train_gen: any,
    val_gen: any,
    epochs: int,
    model_name: str,
) -> keras.callbacks.History:
    """Fit the model and return its training history."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION & PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(history: keras.callbacks.History, model_name: str) -> None:
    """4-panel plot: loss, accuracy, precision, AUC over epochs."""
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    metrics = [
        ("loss", "val_loss", "Loss"),
        ("accuracy", "val_accuracy", "Accuracy"),
        ("precision", "val_precision", "Precision"),
        ("auc", "val_auc", "AUC"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (train_key, val_key, title) in zip(axes, metrics):
        ax.plot(epochs_range, hist.get(train_key, []), label="Train", linewidth=2)
        ax.plot(epochs_range, hist.get(val_key, []), label="Validation",
                linewidth=2, linestyle="--")
        ax.set_title(f"{model_name} — {title}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"Training History: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    fig.savefig(MODELS_DIR / f"{safe_name}_training_history.png", dpi=120)
    plt.close(fig)
    print(f"[Saved] {safe_name}_training_history.png")


def evaluate_model(
    model: keras.Model,
    val_gen: any,
    model_name: str,
) -> Dict:
    """
    Compute and save:
    - Classification report
    - Confusion matrix
    - ROC curve
    Returns a dict of key metrics.
    """
    val_gen.reset()
    y_true = val_gen.classes
    y_scores = model.predict(val_gen, verbose=0).flatten()
    y_pred = (y_scores >= 0.5).astype(int)

    # --- Classification report ---
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    print(f"\n[{model_name}] Classification Report:\n{report}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    fig.savefig(MODELS_DIR / f"{safe_name}_confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"[Saved] {safe_name}_confusion_matrix.png")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / f"{safe_name}_roc_curve.png", dpi=120)
    plt.close(fig)
    print(f"[Saved] {safe_name}_roc_curve.png")

    # Extract scalar metrics
    results = model.evaluate(val_gen, verbose=0)
    metric_names = model.metrics_names
    metric_dict = dict(zip(metric_names, results))
    metric_dict["roc_auc"] = roc_auc
    return metric_dict


def plot_model_comparison(metrics_a: Dict, metrics_b: Dict) -> None:
    """Side-by-side bar chart comparing Model A vs Model B on key metrics."""
    keys = ["accuracy", "precision", "recall", "auc"]
    labels = ["Accuracy", "Precision", "Recall", "AUC"]
    vals_a = [metrics_a.get(k, 0) for k in keys]
    vals_b = [metrics_b.get(k, 0) for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_a = ax.bar(x - width / 2, vals_a, width, label="Custom CNN", color="#3498db")
    bars_b = ax.bar(x + width / 2, vals_b, width, label="MobileNetV2", color="#e74c3c")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Custom CNN vs MobileNetV2", fontsize=13, fontweight="bold")
    ax.legend()

    for bar in bars_a + bars_b:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(MODELS_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)
    print("[Saved] model_comparison.png")

    # Print comparison table
    print("\n" + "="*55)
    print(f"{'Metric':<15} {'Custom CNN':>15} {'MobileNetV2':>15}")
    print("-"*55)
    for lbl, va, vb in zip(labels, vals_a, vals_b):
        print(f"{lbl:<15} {va:>15.4f} {vb:>15.4f}")
    print("="*55)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  MALARIA DETECTION — DEEP LEARNING TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Validate & explore dataset ────────────────────────────────────────
    print("\n[Step 1] Validating dataset ...")
    validate_dataset()

    paths, labels = collect_image_paths()
    print(f"\n[Dataset] Total images: {len(paths)}")
    summarize_image_dimensions(paths)
    plot_class_distribution(labels)
    plot_sample_images(paths, labels)

    # ── 2. Build data generators (80/20 split via validation_split) ───────────
    print("\n[Step 2] Building data generators ...")
    train_gen, val_gen, n_train, n_val = build_generators_split()

    # ── 3. Model A — Custom CNN ───────────────────────────────────────────────
    print("\n[Step 3] Building Custom CNN ...")
    cnn_model = build_cnn_model()
    cnn_model.summary()

    history_cnn = train_model(cnn_model, train_gen, val_gen, EPOCHS_CNN, "Custom_CNN")
    plot_training_history(history_cnn, "Custom CNN")
    train_gen.reset(); val_gen.reset()
    metrics_cnn = evaluate_model(cnn_model, val_gen, "Custom CNN")

    # ── 4. Model B — MobileNetV2 (Phase 1: frozen base) ─────────────────────
    print("\n[Step 4] Building MobileNetV2 (Phase 1 — frozen base) ...")
    mb_model = build_mobilenetv2_model()
    mb_model.summary()

    train_gen.reset(); val_gen.reset()
    history_mb1 = train_model(mb_model, train_gen, val_gen, EPOCHS_TRANSFER, "MobileNetV2_P1")
    plot_training_history(history_mb1, "MobileNetV2 Phase1")

    # ── 5. Model B — Fine-tuning (Phase 2) ───────────────────────────────────
    print("\n[Step 5] Fine-tuning MobileNetV2 (Phase 2) ...")
    mb_model = unfreeze_top_layers(mb_model, n_layers=30)

    train_gen.reset(); val_gen.reset()
    history_mb2 = train_model(mb_model, train_gen, val_gen, FINE_TUNE_EPOCHS, "MobileNetV2_P2")
    plot_training_history(history_mb2, "MobileNetV2 Phase2")

    train_gen.reset(); val_gen.reset()
    metrics_mb = evaluate_model(mb_model, val_gen, "MobileNetV2")

    # ── 6. Compare and select best model ─────────────────────────────────────
    print("\n[Step 6] Comparing models ...")
    plot_model_comparison(metrics_cnn, metrics_mb)

    cnn_score = metrics_cnn.get("auc", 0)
    mb_score = metrics_mb.get("auc", 0)
    best_model_path = str(MODELS_DIR / "best_malaria_model.h5")

    if mb_score >= cnn_score:
        winner = "MobileNetV2"
        mb_model.save(best_model_path)
    else:
        winner = "Custom CNN"
        cnn_model.save(best_model_path)

    print(f"\n{'='*60}")
    print(f"  ✅ Best Model: {winner} (AUC = {max(cnn_score, mb_score):.4f})")
    print(f"  💾 Saved to: {best_model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
