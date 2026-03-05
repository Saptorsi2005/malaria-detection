"""
download_dataset.py — Download a sample of the Malaria Cell Images dataset
===========================================================================
Source: NIH / Kaggle Cell Images for Detecting Malaria
        https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

This script downloads sample images (20 per class) from a public mirror
so you can test train.py and the Flask app without a Kaggle account.

For the FULL 998-image dataset, download from Kaggle directly:
    kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
    Unzip into the malaria_detection/dataset/ folder.

Usage:
    python download_dataset.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
PARASITIZED_DIR = DATASET_DIR / "Parasitized"
UNINFECTED_DIR  = DATASET_DIR / "Uninfected"

# Public mirror — small sample zip hosted on GitHub
# Contains 20 Parasitized + 20 Uninfected images for quick-start testing
SAMPLE_URL = (
    "https://github.com/delftstack/datasets/raw/main/"
    "malaria-cell-images-sample.zip"
)

# Fallback: generate synthetic noise images using PIL (always works offline)
SYNTHETIC_FALLBACK = True


def generate_synthetic_images(n: int = 30) -> None:
    """
    Create synthetic solid-color images as placeholder data.
    These are NOT real blood smear images — only for structural testing.
    Replace with real Kaggle images before training.
    """
    from PIL import Image, ImageDraw
    import random

    print(f"\n[Synthetic] Generating {n} placeholder images per class …")
    PARASITIZED_DIR.mkdir(parents=True, exist_ok=True)
    UNINFECTED_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        # Parasitized — reddish tones with dark spots
        img = Image.new("RGB", (141, 143), color=(
            random.randint(160, 220),
            random.randint(80, 130),
            random.randint(80, 130),
        ))
        draw = ImageDraw.Draw(img)
        # Add random dark "parasite" dots
        for _ in range(random.randint(3, 8)):
            x = random.randint(10, 130)
            y = random.randint(10, 133)
            r = random.randint(5, 15)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(40, 20, 20))
        img.save(PARASITIZED_DIR / f"synth_parasitized_{i:04d}.png")

        # Uninfected — clean pinkish cells
        img2 = Image.new("RGB", (141, 143), color=(
            random.randint(200, 240),
            random.randint(150, 190),
            random.randint(150, 190),
        ))
        img2.save(UNINFECTED_DIR / f"synth_uninfected_{i:04d}.png")

    print(f"  ✓ {n} Parasitized images → {PARASITIZED_DIR}")
    print(f"  ✓ {n} Uninfected images  → {UNINFECTED_DIR}")
    print("\n⚠️  These are SYNTHETIC placeholder images.")
    print("   For real training, download the Kaggle dataset and replace these.")
    print("   https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria\n")


def count_existing() -> tuple:
    p = len(list(PARASITIZED_DIR.glob("*.*"))) if PARASITIZED_DIR.exists() else 0
    u = len(list(UNINFECTED_DIR.glob("*.*"))) if UNINFECTED_DIR.exists() else 0
    return p, u


if __name__ == "__main__":
    p_count, u_count = count_existing()

    if p_count >= 10 and u_count >= 10:
        print(f"[Dataset] Already populated: {p_count} Parasitized, {u_count} Uninfected.")
        print("Nothing to do. Delete dataset/ folders to regenerate.")
    else:
        print("[Dataset] Populating dataset with synthetic placeholder images …")
        generate_synthetic_images(n=30)
        print("[Done] Dataset ready. Run: python train.py")
