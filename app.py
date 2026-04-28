import os
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
IMG_SIZE     = 120                              # must match training
CATEGORIES   = ["NIKE", "ADIDAS"]              # index 0 = NIKE, 1 = ADIDAS
ALLOWED_EXT  = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_MB       = 8

# ── Unrelated-image detection thresholds ──────
# If the winning class probability is below this, the model is "unsure" → unrelated
CONFIDENCE_THRESHOLD = 0.70   # 70% — tune up/down as needed (0.65–0.80 is a good range)
# Shannon entropy of a perfectly uncertain binary output = ln(2) ≈ 0.693
# We flag as unrelated when entropy is HIGH (model can't decide)
ENTROPY_THRESHOLD    = 0.60   # bits — above this = confused model → unrelated

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR   = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# ─────────────────────────────────────────────
#  LOAD MODEL ONCE (warm-up included)
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "nike_adidas_model.keras")
try:
    model = load_model(MODEL_PATH)
    # warm-up: eliminates TF graph-build latency on first real request
    _warm = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    model.predict(_warm, verbose=0)
    log.info("✅  Model loaded and warmed up → %s", MODEL_PATH)
except Exception as e:
    model = None
    log.error("❌  Failed to load model: %s", e)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess(path: str) -> np.ndarray:
    """
    EXACT pipeline used in the training notebook:
      1. Open → convert to Grayscale ('L')
      2. Resize to (IMG_SIZE, IMG_SIZE)
      3. Convert to numpy array
      4. Reshape to (1, IMG_SIZE, IMG_SIZE, 1)
    No extra normalisation – model was trained without it.
    """
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))          # default NEAREST, same as notebook
    arr = np.array(img)                              # shape (120, 120)
    arr = arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)     # shape (1, 120, 120, 1)
    return arr


def shannon_entropy(probs: np.ndarray) -> float:
    """Binary Shannon entropy (nats). Max = ln(2) ≈ 0.693 when probs = [0.5, 0.5]."""
    probs = np.clip(probs, 1e-9, 1.0)              # avoid log(0)
    return float(-np.sum(probs * np.log(probs)))


def run_prediction(path: str) -> dict:
    arr    = preprocess(path)
    probs  = model.predict(arr, verbose=0)[0]        # shape (2,) – softmax outputs
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])                        # 0.0 – 1.0
    entropy = shannon_entropy(probs)

    log.info("Softmax → %s  |  conf=%.3f  entropy=%.3f", probs.tolist(), conf, entropy)

    # ── Unrelated-image guard ────────────────────────────────────────────
    # A genuine Nike/Adidas image produces HIGH confidence + LOW entropy.
    # An unrelated image confuses the binary model → LOW confidence OR HIGH entropy.
    is_unrelated = (conf < CONFIDENCE_THRESHOLD) or (entropy > ENTROPY_THRESHOLD)

    return {
        "unrelated":   is_unrelated,
        "brand":       CATEGORIES[idx],
        "confidence":  round(conf * 100, 2),
        "nike_pct":    round(float(probs[0]) * 100, 2),
        "adidas_pct":  round(float(probs[1]) * 100, 2),
        "entropy":     round(entropy, 4),
    }

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """AJAX endpoint – returns JSON so UI can animate without page reload."""
    if model is None:
        return jsonify({"error": "Model unavailable – check server logs."}), 503

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400

    if not allowed(file.filename):
        return jsonify({"error": f"File type not supported. Use: {', '.join(ALLOWED_EXT)}"}), 415

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = run_prediction(filepath)
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    result["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result)


@app.errorhandler(413)
def file_too_large(_):
    return jsonify({"error": f"File exceeds {MAX_MB} MB limit."}), 413

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)