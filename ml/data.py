"""CSV manifest loader with explicit source and identity leakage checks."""
from __future__ import annotations
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset

REQUIRED = {"path", "label", "split", "source", "group"}
SPLITS = {"train", "val", "test"}


def read_manifest(csv_path, verify_hashes=False):
    manifest = Path(csv_path).resolve()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {', '.join(sorted(REQUIRED))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no images")
    seen_paths, groups, digests = set(), defaultdict(set), defaultdict(set)
    for row in rows:
        row["path"] = str((manifest.parent / row["path"]).resolve())
        if not Path(row["path"]).is_file():
            raise ValueError(f"Missing image: {row['path']}")
        if row["path"] in seen_paths:
            raise ValueError(f"Repeated file: {row['path']}")
        seen_paths.add(row["path"])
        if row["label"] not in {"real", "fake"} or row["split"] not in SPLITS:
            raise ValueError("Each label must be real/fake and split must be train/val/test")
        if not row["source"].strip() or not row["group"].strip():
            raise ValueError("Every row needs a source and group identifier")
        groups[row["group"]].add(row["split"])
        if verify_hashes:
            digest = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            digests[digest].add(row["split"])
    overlap = [key for key, splits in groups.items() if len(splits) > 1]
    if overlap:
        raise ValueError(f"Identity/group leakage across splits: {overlap[:5]}")
    duplicate = [key for key, splits in digests.items() if len(splits) > 1]
    if duplicate:
        raise ValueError("Duplicate image bytes appear across splits")
    return rows


class ImageManifestDataset(Dataset):
    def __init__(self, rows, split, transform):
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} samples found")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        return self.transform(image), float(row["label"] == "fake"), row["source"]
