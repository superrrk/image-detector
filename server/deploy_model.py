"""Fetch a public checkpoint at boot, without storing model weights in Git.

Set MODEL_URL to an HTTPS download URL and optionally MODEL_SHA256 to its
SHA-256 digest. Fails startup if no checkpoint can be provisioned.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "image_origin_resnet18_v2.pth"
MAX_CHECKPOINT_BYTES = 125 * 1024 * 1024


def ensure_checkpoint():
    target = Path(os.environ.get("MODEL_PATH", str(DEFAULT_PATH)))
    expected = os.environ.get("MODEL_SHA256", "").lower().strip()
    if expected and (len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("MODEL_SHA256 must be a 64-character hex string")

    if target.is_file():
        if not expected or _sha256(target) == expected:
            print(f"Checkpoint ready: {target}", flush=True)
            return target
        raise RuntimeError("Existing model checksum does not match MODEL_SHA256")

    url = os.environ.get("MODEL_URL", "").strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise RuntimeError("MODEL_URL must be set to a valid HTTPS checkpoint download URL")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    sha = hashlib.sha256()
    count = 0
    try:
        with urlopen(Request(url, headers={"User-Agent": "image-forensics-lab/1.0"}), timeout=90) as response:
            if urlparse(response.geturl()).scheme != "https":
                raise RuntimeError("Model download redirected to non-HTTPS URL")
            with temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > MAX_CHECKPOINT_BYTES:
                        raise RuntimeError("Model exceeds the 125 MB checkpoint limit")
                    sha.update(chunk)
                    out.write(chunk)
        if count < 1024:
            raise RuntimeError("Model download is too small to be a checkpoint")
        if expected and sha.hexdigest() != expected:
            raise RuntimeError("Model checksum mismatch: verify MODEL_URL and MODEL_SHA256")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded model checkpoint ({count:,} bytes): {target}", flush=True)
    return target


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    ensure_checkpoint()
