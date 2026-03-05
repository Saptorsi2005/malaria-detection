"""
Malaria Detection — Flask Web Application
==========================================
Routes:
    GET  /            → Upload page  (index.html)
    POST /predict     → Run inference, redirect to result.html
    GET  /health      → Health-check JSON endpoint

Run locally:
    flask --app app run --debug

Production:
    gunicorn -w 4 -b 0.0.0.0:8000 app:app
"""

import os
import uuid
import logging
from pathlib import Path

from flask import (
    Flask, request, render_template, redirect,
    url_for, flash, jsonify,
)
from werkzeug.utils import secure_filename

# predict is imported lazily inside routes so TensorFlow is NOT loaded at
# Flask startup — this avoids the "Aborted!" crash when keras is absent/wrong.

# ─── Application Setup ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "malaria-detection-secret-key-change-in-prod")

# ─── Upload Configuration ─────────────────────────────────────────────────────
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}
MAX_CONTENT_LENGTH_MB = 10
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024  # 10 MB

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Return True if the file's extension is in the allowed set."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_obj) -> Path:
    """
    Save an uploaded file with a UUID-prefixed name to the uploads folder.
    Returns the saved file path.
    """
    original_name = secure_filename(file_obj.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = UPLOAD_FOLDER / unique_name
    file_obj.save(str(save_path))
    logger.info(f"Saved upload: {save_path}")
    return save_path


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Render the main upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accept an uploaded image, run malaria detection, and display result.
    """
    # ── Validate request ──────────────────────────────────────────────────────
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose an image.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(
            "Unsupported file type. Please upload a PNG, JPG, or BMP image.",
            "error",
        )
        return redirect(url_for("index"))

    # ── Save, predict, and respond ────────────────────────────────────────────
    saved_path = None
    try:
        saved_path = save_upload(file)

        # Lazy import — TensorFlow loads only on first prediction request
        from predict import predict_from_path
        result = predict_from_path(str(saved_path))

        rel_path = saved_path.relative_to(BASE_DIR)
        result["image_url"] = "/" + str(rel_path).replace("\\", "/")

        logger.info(
            f"Prediction: {result['label']} ({result['confidence']:.2f}%) "
            f"| File: {saved_path.name}"
        )
        return render_template("result.html", result=result)

    except FileNotFoundError:
        flash(
            "⚠️ No trained model found. Please run 'python train.py' first to train the model.",
            "error",
        )
        return redirect(url_for("index"))

    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        flash(f"Prediction failed: {str(exc)}", "error")
        return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for programmatic access.
    Accepts multipart/form-data with a 'file' field.
    Returns JSON with label, confidence, raw_score.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 415

    try:
        saved_path = save_upload(file)
        from predict import predict_from_path
        result = predict_from_path(str(saved_path))
        return jsonify({
            "prediction": result["label"],
            "confidence": result["confidence"],
            "raw_score": result["raw_score"],
            "status": "success",
        })
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run train.py first.", "status": "failed"}), 503
    except Exception as exc:
        logger.error(f"API prediction failed: {exc}", exc_info=True)
        return jsonify({"error": str(exc), "status": "failed"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint for deployment monitoring."""
    return jsonify({"status": "ok", "service": "Malaria Detection API"})


# Serve uploaded images so the result page can display them
from flask import send_from_directory

@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    flash(f"File too large. Maximum allowed size is {MAX_CONTENT_LENGTH_MB} MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
