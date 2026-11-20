
from datasets import load_dataset
from pathlib import Path
from PIL import Image

print("Connecting to Hugging Face...", flush=True)

dataset = load_dataset(
    "elsaEU/ELSA_D3",
    split="train",
    streaming=True
)

output = Path("dataset")
limits = {"train": 800, "val": 100, "test": 100}

saved = 0

for index, row in enumerate(dataset):
    if saved >= 1000:
        break

    image = row.get("image_gen0")

    if not isinstance(image, Image.Image):
        continue

    if saved < 800:
        split = "train"
    elif saved < 900:
        split = "val"
    else:
        split = "test"

    folder = output / split / "fake"
    folder.mkdir(parents=True, exist_ok=True)

    filename = folder / f"fake_{saved:05d}.jpg"
    image.convert("RGB").save(filename, quality=95)

    saved += 1

    if saved % 100 == 0:
        print(f"Downloaded {saved}/1000 images", flush=True)

print(f"Finished! Saved {saved} images.")
