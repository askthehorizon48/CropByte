import json
import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "cropbyte_model.tflite"
LABELS_PATH = ROOT / "cropbyte_labels.json"
METADATA_PATH = ROOT / "cropbyte_model.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ASSETS = {"index.html", "main.html", "style.css", "script.js", "translation.js", "README.md"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 1024 * 1024
inference_lock = Lock()


def format_label(label):
    return re.sub(r"\s+", " ", label.replace("___", " - ").replace("_", " ")).strip()


@lru_cache(maxsize=1)
def load_classifier():
    for path in (MODEL_PATH, LABELS_PATH, METADATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                "The classifier is not trained yet. Add the 15 supported "
                "PlantVillage class folders under data\\PlantVillage, then run "
                "the training command shown on the page."
            )

    with LABELS_PATH.open(encoding="utf-8") as labels_file:
        labels = json.load(labels_file)
    if not isinstance(labels, list) or not labels:
        raise ValueError("The model labels file must contain a non-empty JSON array.")
    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("preprocessing") != "mobilenet_v2_preprocess_input":
        raise ValueError(
            "This model uses incompatible image preprocessing. "
            "Retrain it by following README.md."
        )
    if metadata.get("classes") != labels:
        raise ValueError("Model metadata and cropbyte_labels.json do not match.")
    if metadata.get("input_size") != 224:
        raise ValueError("Model metadata has an unsupported image input size.")

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    expected_input = (1, metadata["input_size"], metadata["input_size"], 3)
    if tuple(int(value) for value in input_details["shape"]) != expected_input:
        raise ValueError(
            f"Model input shape {input_details['shape']} does not match metadata "
            f"{expected_input}. Retrain the model."
        )
    if int(output_details["shape"][-1]) != len(labels):
        raise ValueError(
            f"Model returns {output_details['shape'][-1]} classes but "
            f"{LABELS_PATH.name} contains {len(labels)}."
        )
    return interpreter, labels, input_details, output_details


def prepare_input(image, input_details):
    _, height, width, channels = (int(value) for value in input_details["shape"])
    if channels != 3:
        raise ValueError(f"Expected a three-channel image input, got {channels} channels.")
    pixels = np.asarray(image.resize((width, height)), dtype=np.float32)
    pixels = pixels / 127.5 - 1.0
    dtype = input_details["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = input_details["quantization"]
        if scale == 0:
            raise ValueError("The model input has invalid quantization parameters.")
        limits = np.iinfo(dtype)
        pixels = np.clip(np.round(pixels / scale + zero_point), limits.min, limits.max)
    return np.expand_dims(pixels.astype(dtype), axis=0)


def to_probabilities(output):
    scores = np.asarray(output, dtype=np.float32).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("The model returned invalid prediction scores.")
    if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
        return scores
    shifted = scores - scores.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:asset>")
def frontend_asset(asset):
    if asset not in ASSETS:
        return jsonify(error="Not found"), 404
    return send_from_directory(ROOT, asset)


@app.get("/api/health")
def health():
    try:
        _interpreter, labels, input_details, _output_details = load_classifier()
    except (FileNotFoundError, ValueError) as error:
        return jsonify(ready=False, error=str(error)), 503
    except Exception:
        app.logger.exception("Could not load the crop disease model.")
        return jsonify(ready=False, error="The prediction model could not be loaded."), 503
    return jsonify(
        ready=True,
        classes=len(labels),
        model_input=[int(value) for value in input_details["shape"]],
    )


@app.post("/api/predict")
def predict():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Choose an image before analyzing."), 400

    image_data = uploaded.stream.read(MAX_IMAGE_BYTES + 1)
    if len(image_data) > MAX_IMAGE_BYTES:
        return jsonify(error="Image is too large. Choose an image smaller than 10 MB."), 413

    try:
        with Image.open(BytesIO(image_data)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return jsonify(error="That file is not a readable image. Try a JPG, PNG, or WebP image."), 400

    try:
        interpreter, labels, input_details, output_details = load_classifier()
        model_input = prepare_input(image, input_details)
        with inference_lock:
            interpreter.set_tensor(input_details["index"], model_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
        if np.issubdtype(output_details["dtype"], np.integer):
            scale, zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale
        probabilities = to_probabilities(output)
    except (FileNotFoundError, ValueError) as error:
        app.logger.warning("Crop disease model is not ready: %s", error)
        return jsonify(error=str(error)), 503
    except Exception:
        app.logger.exception("Crop disease prediction failed.")
        return jsonify(error="Analysis failed. Check the model files and server log, then try again."), 500

    best_index = int(np.argmax(probabilities))
    label = labels[best_index]
    return jsonify(
        prediction=format_label(label),
        class_name=label,
        confidence=float(probabilities[best_index]),
        healthy=label.lower().endswith("healthy"),
        alternatives=[
            {"prediction": format_label(labels[index]), "confidence": float(probabilities[index])}
            for index in np.argsort(probabilities)[::-1][1:3]
        ],
    )


@app.errorhandler(413)
def image_too_large(_error):
    return jsonify(error="Image is too large. Choose an image smaller than 10 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
