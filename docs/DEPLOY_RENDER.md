# Deploy Image Forensics Lab on Render

## Required before starting

- A GitHub repository containing the application code, `render.yaml`, `Dockerfile`, `requirements-deploy.txt`, `server/` and `ml/`.
- A copy of the **v2** trained checkpoint (`image_origin_resnet18_v2.pth`). Do not deploy v3: the pilot metrics were worse.
- A public HTTPS download location for the checkpoint, **if you have the right to redistribute it**. A GitHub Release asset can work; do not add training data or private photographs to Git.
- A Render account and an explicit choice of a compute plan. `render.yaml` specifies `1c-2g` (2 GB RAM), **a paid plan**. Free (512 MB) may run out of memory with PyTorch.

## 1. Publish code to the existing GitHub repo

Copy the deployment files into `image-detector-publish` (do not overwrite its `.git` directory). Commit, push your `portfolio-upgrade` branch and merge into `main` only after checking that no datasets, private images, or weights are staged.

## 2. Host the checkpoint separately

If redistribution is permitted, upload the checkpoint as a GitHub Release asset on `superrrk/image-detector`:

1. Open GitHub repository → Releases → **Draft a new release**.
2. Tag it `model-v2`, attach `image_origin_resnet18_v2.pth`, and publish.
3. Copy the asset's **actual** HTTPS download link (don't use the release page's browser URL).
4. Locally calculate its SHA-256:
   `shasum -a 256 ~/Downloads/image-forensics-lab/models/image_origin_resnet18_v2.pth`

Alternatively use another trusted static HTTPS file host. Keep URL and hash together.

## 3. Create a Render Blueprint

On Render, choose **New → Blueprint**, connect `superrrk/image-detector` and select the branch containing `render.yaml`. Render reads `render.yaml` and provisions one Docker web service.

When prompted, fill:
- `MODEL_URL`: actual HTTPS checkpoint asset download URL
- `MODEL_SHA256`: hash from Step 2 (recommended)

**Review paid-plan charges before approving**. The Blueprint specifies `plan: 1c-2g` (2 GB RAM). Do not assume this is free.

The Docker container installs CPU-only PyTorch, downloads the model at startup, and serves both UI and API with Gunicorn. No separate Vercel or Netlify frontend is needed.

## 4. Verify the live site

When Render says Live, take the URL that Render actually assigns (do not guess it) and run:

```bash
SITE=https://your-actual-render-url.onrender.com
curl -i "$SITE/"
curl -i "$SITE/api/health"
curl -i "$SITE/api/ready"
curl -i -F "image=@example.jpg" "$SITE/api/analyze"
```

Expected: `/` HTTP 200; `/api/ready` HTTP 200 with `model_available: true`; `/api/analyze` HTTP 200 with non-null `prediction` (not just inspection stats). Finally test drag/drop, paste, Remove and Rotate in Chrome and Safari.

## Troubleshooting

- `/api/ready` 503: incorrect/missing model URL, corrupt checkpoint, mismatched model architecture, or missing PyTorch.
- Docker image too big: ensure only CPU wheels installed, and that `.dockerignore` excludes `dataset/` and `models/`.
- Out of memory or exit 137: change to a plan with enough RAM; keep one Gunicorn worker.
- Upload 413: max 8 MB per image, 12 million decoded pixels.
- Browser 404: check the actual Render URL and ensure `server.app:app` is running.
- Cold start/download errors: check startup logs for checkpoint URL and checksum verification.
- The model still misclassifies some generated images; the site must label predictions experimental and not proof of authenticity.
