# Image Forensics Lab

A functional, privacy-conscious image-inspection web application with an **optional** experimental PyTorch classifier. Built with Python, Flask, Pillow and NumPy, with a no-Flask local fallback.

**This is not an AI-authenticity verifier.** Image metadata, JPEG recompression differences, and pixel statistics are descriptive measurements. They do not establish whether an image is real or AI-generated. The model output, when available, is an uncalibrated experimental score, not proof or a confidence probability.

## Quick start

### Option A — lightweight local demo (no Flask or model needed)

Install Python 3.11+ and dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pillow numpy
python -m server.standalone
```

Open **http://127.0.0.1:5000**.

### Option B — Flask application (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m server.app
```

Open **http://127.0.0.1:5000**. It works **without the model checkpoint** for all image-inspection features.

On Windows, activate with `.venv\\Scripts\\activate` and use `python` instead of `python3`.

### Optional legacy CNN predictions

The original repository does **not** include the trained model weights. To enable predictions, install PyTorch with `pip install -r requirements-ml.txt` and place a compatible **legacy CNN** `state_dict` at `models/fake_detector.pth`, or set `MODEL_PATH` to an absolute path.

The old `train_improved.py` checkpoints have a different architecture and preprocessing, so **are not compatible** with the legacy loader. Do not download or execute unknown checkpoints from untrusted sources. When no compatible model is installed, the app clearly labels this as unavailable rather than fabricating a prediction.

## Features

- Browser-based drag and drop and responsive UI, no frontend build step.
- Actual PNG, JPEG, or WebP image decoding with EXIF orientation correction.
- Image dimensions, file size, and presence of EXIF metadata (does not expose private metadata values).
- Brightness, contrast, grayscale entropy, sharpness (Laplacian variance), mean color, and JPEG recompression error.
- Visual JPEG recompression difference map (ELA-style, **not** an authenticity map).
- Optional CNN classification score and gradient sensitivity overlay.
- Images processed entirely in memory (no uploads or history are persisted).
- 8 MB file limit and 12-megapixel pixel limit; malformed input handling.
- Flask integration tests and pure inspection tests; GitHub Actions CI.
- Dockerized Flask deployment with a production WSGI server.

## Routes

| Endpoint | Description |
| --- | --- |
| `GET /` | Browser UI |
| `GET /api/health` | Server status and whether model checkpoint file exists |
| `POST /api/analyze` | Accepts multipart file field `image`; returns JSON inspection, optional prediction, and embedded visualizations |

Example:

```bash
curl -F 'image=@sample.jpg' http://127.0.0.1:5000/api/analyze
```

## Engineering decisions

- **Failure isolation:** image inspection succeeds even when optional model weights are absent.
- **Stateless processing:** no disk writes or persistent uploads; analysis visuals are embedded in response payloads.
- **Unambiguous outputs:** measured image properties are separated from any ML prediction.
- **Testability:** `create_app(predictor=...)` injects test doubles, so API tests do not require a model or GPU.
- **Consistency:** optional model code is shared by Flask and the local fallback server.
- **Guardrails:** restricted image formats and dimensions, non-debug Flask server, and no wildcard CORS.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Deployment

Build and run the Flask container:

```bash
docker build -t image-forensics-lab .
docker run --rm -p 8080:8080 image-forensics-lab
```

Open `http://127.0.0.1:8080`. This intentionally deploys the useful image inspection service without ~gigabytes of GPU/ML packages. If you deploy the optional classifier, you must install compatible PyTorch dependencies and provide model weights separately. For an internet-facing service, put upload rate limits, HTTPS, request timeouts and operational monitoring in front of the container.

## What still needs verification to make a strong ML claim

- Recover or retrain model weights with an identifiable training dataset and class mapping.
- Ensure splits separate identities and generator families, avoiding leakage.
- Record train/validation/test sample counts and source licenses.
- Compare to baselines using held-out precision, recall, F1, confusion matrix, and ROC-AUC.
- Measure inference latency and performance under distribution shift.
- Evaluate calibration before using the word *confidence*.

The previous README reported **87% validation accuracy**. That historical claim is **not reproduced or independently verified** by this package and should not be described as a new benchmark result.

## Portfolio description

> Built a privacy-conscious image-inspection web application using Flask, Pillow, and NumPy; implemented in-memory upload validation, pixel-level forensic visualizations, an optional PyTorch inference pipeline, automated API tests, and containerized deployment.

Only include the features you've actually deployed and tested when describing your work to recruiters.

## One-command launch after installing Pillow + NumPy

On macOS/Linux, use `./start.sh`. On Windows, use `start.bat`.

## Adoption in the original repository

See [`docs/INTEGRATE.md`](docs/INTEGRATE.md) for instructions that preserve older research scripts and avoid deleting committed datasets or the React submodule.

## New: Model v2 training pipeline

A **new ResNet18-based classifier, reproducible training script, manifest validator, validation-only threshold selection, and per-generator test reporting** are included in [`docs/MODEL_V2.md`](docs/MODEL_V2.md). This replaces neither your dataset nor the old weights automatically. There is **no new measured accuracy** until you train and evaluate it on labeled, independent real/fake examples.

```bash
pip install -r requirements-train.txt
python -m ml.prepare_manifest --root path/to/dataset --output dataset/manifest.csv
python -m ml.train --manifest dataset/manifest.csv --pretrained --epochs 8 --verify-hashes
MODEL_PATH=models/image_origin_resnet18_v2.pth python -m server.app
```
