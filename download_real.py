
from datasets import load_dataset
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from urllib.parse import urlparse
import requests
import time

dataset = load_dataset(
    "elsaEU/ELSA_D3",
    split="train",
    streaming=True
)

session = requests.Session()
saved = 0
matched = 0

print("Downloading matching real photographs...", flush=True)

for row in dataset:
    # Match the sequence used by download_dataset.py.
    if matched >= 1000:
        break

    if not isinstance(row.get("image_gen0"), Image.Image):
        continue

    index = matched
    matched += 1

    if index < 800:
        split = "train"
    elif index < 900:
        split = "val"
    else:
        split = "test"

    folder = Path("dataset") / split / "real"
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f"real_{index:05d}.jpg"

    if filename.exists():
        saved += 1
        continue

    url = row.get("url")
    if not isinstance(url, str):
        continue

    if urlparse(url).scheme not in ("http", "https"):
        continue

    try:
        response = session.get(url, timeout=12)
        response.raise_for_status()

        if len(response.content) > 15_000_000:
            continue

        image = Image.open(BytesIO(response.content))
        image.load()

        if image.width < 128 or image.height < 128:
            continue

        image = image.convert("RGB")
        image.save(filename, quality=95)
        saved += 1

        if saved % 50 == 0:
            print(f"Saved {saved} real images", flush=True)

    except (requests.RequestException, OSError,
            UnidentifiedImageError) as error:
        print(f"Skipping {index}: {str(error)[:80]}")

    time.sleep(0.1)

print(f"Finished! Saved {saved} real images.")
