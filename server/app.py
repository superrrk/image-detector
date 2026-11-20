"""Flask frontend/API for real, non-diagnostic image inspection."""
from __future__ import annotations
import io
import logging
import os
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from server.core import MAX_BYTES, MAX_PIXELS, ModelUnavailable, Predictor
from server.forensics import inspect_image
logger = logging.getLogger(__name__)

def create_app(predictor=None):
    app = Flask(__name__, static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    engine = predictor or Predictor()

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model_available=bool(getattr(engine, "checkpoint", None) and engine.checkpoint.is_file()),
                       checkpoint_name=engine.checkpoint.name if getattr(engine, "checkpoint", None) and engine.checkpoint.is_file() else None)

    @app.get("/api/ready")
    def ready():
        # A deployment is not ready unless inference can actually load the weights.
        try:
            engine._load()
        except ModelUnavailable:
            logger.exception("Model checkpoint unavailable during readiness check")
            return jsonify(status="unavailable", model_available=False), 503
        except Exception:
            logger.exception("Model initialization failed during readiness check")
            return jsonify(status="error", model_available=False), 503
        return jsonify(status="ready", model_available=True)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Image exceeds the 8 MB upload limit."), 413

    @app.post("/api/analyze")
    def analyze():
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify(error="Choose an image to analyze."), 400
        try:
            # Browser MIME metadata is untrusted; verify actual decoded image bytes.
            raw = file.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                return jsonify(error="Image exceeds the 8 MB upload limit."), 413
            with Image.open(io.BytesIO(raw)) as opened:
                source_format = opened.format
                exif_present = bool(opened.getexif())
                if opened.format not in {"JPEG", "PNG", "WEBP"}:
                    return jsonify(error="Only JPEG, PNG and WebP images are supported."), 415
                if opened.width * opened.height > MAX_PIXELS:
                    return jsonify(error="Image dimensions are too large."), 413
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="That file is not a valid supported image."), 415
        try:
            rotation = int(request.form.get("rotation", "0"))
            if rotation not in (0, 90, 180, 270):
                return jsonify(error="Invalid image rotation."), 400
            if rotation:
                image = image.rotate(-rotation, expand=True)
        except (ValueError, TypeError):
            return jsonify(error="Invalid image rotation."), 400
        try:
            inspection = inspect_image(image, source_format=source_format,
                                       byte_count=len(raw), exif_present=exif_present)
        except Exception:
            logger.exception("Image inspection failed")
            return jsonify(error="Image inspection failed."), 500
        try:
            inspection["prediction"] = engine.predict(image)
            inspection["model_status"] = "ready"
        except ModelUnavailable:
            # The forensic inspection remains useful without optional weights.
            inspection["prediction"] = None
            inspection["model_status"] = "not_installed"
        except Exception:
            logger.exception("Optional model inference failed")
            inspection["prediction"] = None
            inspection["model_status"] = "error"
        return jsonify(inspection)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
