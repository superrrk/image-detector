"""Non-diagnostic image inspection; works without an ML model or network access."""
from __future__ import annotations

import io
import math
from collections import Counter

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat


def _finite(number: float, decimals: int = 2) -> float:
    value = float(number)
    return round(value, decimals) if math.isfinite(value) else 0.0


def inspect_image(image: Image.Image, *, source_format: str, byte_count: int, exif_present: bool) -> dict:
    """Return descriptive image signals, not authenticity scores or probabilities."""
    width, height = image.size
    # Bound analysis cost regardless of uploaded dimensions.
    sample = image.copy()
    sample.thumbnail((512, 512), Image.Resampling.LANCZOS)
    rgb = np.asarray(sample.convert('RGB'), dtype=np.float32)
    gray = np.asarray(sample.convert('L'), dtype=np.float32)
    luminance = float(gray.mean())
    contrast = float(gray.std())
    channels = rgb.mean(axis=(0, 1))
    # Laplacian is a descriptive focus/high-frequency indicator, not a detector.
    padded = np.pad(gray, 1, mode='edge')
    laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                 + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * gray)
    sharpness = float(laplacian.var())
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(p * np.log2(p)).sum())
    # Error level analysis visualizes variation after JPEG recompression. Because
    # JPEG recompression *itself* creates errors, this cannot authenticate an image.
    buffer = io.BytesIO()
    sample.save(buffer, format='JPEG', quality=88, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        difference = ImageChops.difference(sample.convert('RGB'), recompressed.convert('RGB'))
        error = float(np.asarray(difference, dtype=np.float32).mean())
        extrema = difference.getextrema()
        max_error = max(channel[1] for channel in extrema)
        visibility = 255.0 / max(1, max_error)
        enhanced = difference.point(lambda x: min(255, int(x * visibility)))
        # A small, inline illustration for immediate demo usability.
        encoded = io.BytesIO()
        enhanced.save(encoded, format='PNG')
    import base64
    ela_url = 'data:image/png;base64,' + base64.b64encode(encoded.getvalue()).decode('ascii')
    return {
        'file': {
            'format': source_format,
            'width': width,
            'height': height,
            'megapixels': _finite(width * height / 1_000_000),
            'bytes': byte_count,
            'exif_present': bool(exif_present),
        },
        'metrics': {
            'brightness': _finite(luminance, 1),
            'contrast': _finite(contrast, 1),
            'sharpness_laplacian_variance': _finite(sharpness, 1),
            'grayscale_entropy_bits': _finite(entropy, 2),
            'jpeg_recompression_mean_error': _finite(error, 2),
            'mean_rgb': [round(float(v)) for v in channels],
        },
        'ela_data_url': ela_url,
        'interpretation': 'These are descriptive properties, NOT evidence that a picture is real, manipulated, or AI-generated. Error-level analysis is especially sensitive to file format and recompression.',
    }
