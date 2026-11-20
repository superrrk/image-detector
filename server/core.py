"""Shared model and inference code; Flask is optional."""
from __future__ import annotations
import base64
import io
import logging
import os
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
logger = logging.getLogger(__name__)

class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    """Load a legacy checkpoint once, on demand, and expose explicit model scores."""

    def __init__(self, checkpoint=None):
        self.checkpoint = Path(checkpoint or os.getenv("MODEL_PATH") or (ROOT / "models" / "image_origin_resnet18_v2.pth" if (ROOT / "models" / "image_origin_resnet18_v2.pth").is_file() else ROOT / "models" / "fake_detector.pth"))
        self._model = None
        self._threshold = 0.5
        self._torch = None
        self._architecture = "legacy-cnn-v1"

    def _load(self):
        if self._model is not None:
            return
        if not self.checkpoint.is_file():
            raise ModelUnavailable("Model weights are not installed. See README for setup.")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ModelUnavailable("PyTorch is not installed.") from exc

        class FakeImageCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(64 * 30 * 30, 128), nn.ReLU(),
                    nn.Linear(128, 1),
                )

            def forward(self, x):
                return torch.sigmoid(self.model(x))

        try:
            # weights_only disallows arbitrary pickle objects in untrusted checkpoints.
            loaded = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            state = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
            if isinstance(loaded, dict) and loaded.get("architecture") == "resnet18-v2":
                if loaded.get("class_to_index") != {"real": 0, "fake": 1}:
                    raise ValueError("Unknown class ordering in checkpoint")
                from ml.model import build_model
                model = build_model(pretrained=False)
                self._architecture = "resnet18-v2"
            else:
                model = FakeImageCNN()
                self._architecture = "legacy-cnn-v1"
            model.load_state_dict(state, strict=True)
            model.eval()
            if isinstance(loaded, dict) and "threshold" in loaded:
                value = float(loaded["threshold"])
                if 0 < value < 1:
                    self._threshold = value
            self._model = model
            self._torch = torch
        except Exception as exc:
            logger.exception("Unable to load model checkpoint")
            raise ModelUnavailable("Model checkpoint is incompatible or could not be loaded.") from exc

    def predict(self, image):
        self._load()
        torch = self._torch
        if self._architecture == "resnet18-v2":
            from ml.model import transform
            tensor = transform(False)(image).unsqueeze(0).requires_grad_(True)
        else:
            prepared = image.resize((128, 128), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(__import__("numpy").asarray(prepared).copy()).permute(2, 0, 1).float() / 255.0
            tensor = pixels.unsqueeze(0).requires_grad_(True)
        output = self._model(tensor).reshape(())
        probability = output.sigmoid() if self._architecture == "resnet18-v2" else output
        score = float(probability.detach().item())
        label = "AI-generated-like" if score >= self._threshold else "Real-like"
        # This is a gradient visualization, not a causal explanation.
        self._model.zero_grad(set_to_none=True)
        probability.backward()
        gradients = tensor.grad.detach().abs().mean(dim=1)[0].numpy()
        maximum = float(gradients.max())
        if maximum > 0:
            gradients = gradients / maximum
        overlay = Image.fromarray((gradients * 255).astype("uint8"), "L")
        # Keep API payloads small even for multi-megapixel photographs.
        base = image.copy()
        base.thumbnail((640, 640), Image.Resampling.LANCZOS)
        overlay = overlay.resize(base.size, Image.Resampling.BILINEAR)
        heat = Image.new("RGB", base.size, (244, 94, 75))
        blend = Image.composite(heat, base, overlay.point(lambda x: int(x * 0.48)))
        buffer = io.BytesIO()
        blend.save(buffer, format="PNG", optimize=True)
        return {
            "label": label,
            "fake_score": round(score, 4),
            "decision_threshold": round(self._threshold, 4),
            "model": self._architecture,
            "saliency_data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "disclaimer": "Experimental model score, not a verified probability or proof of authenticity.",
        }
