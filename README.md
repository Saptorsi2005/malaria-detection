# 🔬 Malaria Detection using Deep Learning

A complete, production-ready deep learning system that classifies microscopic **Giemsa-stained blood smear images** as:

- 🦟 **Parasitized** — Malaria parasite detected  
- ✅ **Uninfected** — Normal red blood cells

Built with **TensorFlow / Keras** (Custom CNN + MobileNetV2 transfer learning) and a **Flask** web interface.

---

## 📁 Project Structure

```
malaria_detection/
├── dataset/
│   ├── Parasitized/     ← place your images here
│   └── Uninfected/      ← place your images here
├── models/              ← saved .h5 files (auto-generated)
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/             ← temp upload files (auto-generated)
├── train.py             ← training pipeline
├── predict.py           ← CLI & API prediction module
├── app.py               ← Flask web application
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Setup

### 1. Clone / Navigate to Project

```bash
cd malaria_detection
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

Download from Kaggle:  
👉 [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

Place images in:

```
dataset/
├── Parasitized/   ← 499 infected cell images
└── Uninfected/    ← 499 uninfected cell images
```

---

## 🏋️ Training

```bash
python train.py
```

This will:
1. Validate and explore the dataset (plots saved to `models/`)
2. Build an **80/20 train/val split** with augmentation
3. Train **Model A** — Custom CNN (40 epochs max with early stopping)
4. Train **Model B** — MobileNetV2 (frozen → fine-tuned)
5. Evaluate both with confusion matrix, ROC curve, classification report
6. Auto-select the best model → saved as `models/best_malaria_model.h5`

Training artifacts saved in `models/`:

| File | Description |
|------|-------------|
| `best_malaria_model.h5` | Best overall model |
| `Custom_CNN_best.h5` | Best checkpoint for CNN |
| `MobileNetV2_P2_best.h5` | Best checkpoint for MobileNetV2 |
| `class_distribution.png` | Dataset balance chart |
| `sample_images.png` | Sample image grid |
| `*_training_history.png` | Loss/accuracy curves |
| `*_confusion_matrix.png` | Confusion matrices |
| `*_roc_curve.png` | ROC curves |
| `model_comparison.png` | Side-by-side comparison |

---

## 🔍 CLI Prediction

```bash
# Default (uses best_malaria_model.h5)
python predict.py path/to/image.png

# Custom model path
python predict.py path/to/image.png --model models/Custom_CNN_best.h5
```

Example output:
```
=======================================================
  MALARIA DETECTION — PREDICTION RESULT
=======================================================
  Image       : test_cell.png
  Prediction  : Parasitized
  Confidence  : 94.27%
  Raw Score   : 0.057302
-------------------------------------------------------
  ⚠️  Malaria parasite detected in the blood sample.
=======================================================
```

---

## 🌐 Flask Web App

### Run Locally (Development)

```bash
flask --app app run --debug
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Run for Production (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### API Usage

```bash
curl -X POST http://localhost:5000/api/predict \
     -F "file=@blood_smear.png"
```

Response:
```json
{
  "prediction": "Parasitized",
  "confidence": 94.27,
  "raw_score": 0.057302,
  "status": "success"
}
```

---

## 🧠 Model Architecture

### Model A — Custom CNN

```
Input (224, 224, 3)
→ [Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)] × 3
→ Flatten → Dense(256) → BN → Dropout(0.5) → Dense(128) → Dropout(0.4)
→ Dense(1, sigmoid)
```

### Model B — MobileNetV2 Transfer Learning

```
Phase 1: MobileNetV2 (frozen) → GlobalAvgPool → Dense(256) → BN → Dropout → Dense(128) → Sigmoid
Phase 2: Unfreeze top 30 layers → fine-tune with lr=1e-5
```

---

## 📊 Expected Performance (998-image dataset)

| Metric | Custom CNN | MobileNetV2 |
|--------|-----------|-------------|
| Accuracy | ~85–90% | ~90–96% |
| AUC | ~0.90 | ~0.95 |
| Inference | < 100ms | < 200ms |

> Actual results depend on dataset composition and random seed.

---

## 🛡️ Overfitting Prevention

- Heavy augmentation (rotation, shift, zoom, flip)
- Batch Normalization in every block
- Dropout (0.25–0.5 at multiple layers)
- EarlyStopping on validation AUC
- ReduceLROnPlateau
- ImageNet transfer weights

---

## ⚕️ Medical Disclaimer

This tool is intended for **educational and research purposes only**.  
It does not constitute a medical diagnosis. Always verify AI predictions with licensed healthcare professionals and proper laboratory tests.

---

## 📜 License & Credits

- Dataset: [Kaggle — Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- Framework: TensorFlow / Keras, Flask
- Built as a Final Year AI/ML Project
