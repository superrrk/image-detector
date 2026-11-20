# CPU-only PyTorch inference image for Render. No training datasets in this image.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=10000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Use official CPU wheels; avoid unnecessary CUDA wheels.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 torchvision==0.21.0
COPY requirements-deploy.txt ./
RUN python -m pip install --no-cache-dir -r requirements-deploy.txt
COPY server ./server
COPY ml ./ml
COPY scripts ./scripts
# MODEL_URL is configured through Render; weights are intentionally excluded from Git.
EXPOSE 10000
CMD ["sh", "-c", "python -m server.deploy_model && exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120 --access-logfile - server.app:app"]
