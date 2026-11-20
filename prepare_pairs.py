"""Prepare a pair-aware ELSA_D3 pilot manifest from existing downloads.

Does not change the original dataset directories. Assumes download_dataset.py
assigned fake_NNNNN indices from image_gen0 and download_real.py used the same
row indices for real_NNNNN files. Exact duplicate bytes are removed.
"""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from PIL import Image

ROOT = Path('dataset')
OUT = ROOT / 'paired_manifest.csv'
SEED = 42


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def main():
    if not ROOT.is_dir():
        raise SystemExit('Missing dataset/ directory. Run from the image-forensics-lab root.')
    pairs = []
    missing = 0
    duplicates = 0
    corrupt = 0
    seen_real = set()
    seen_fake = set()
    # Original downloader used index 0..999, distributed across old directories.
    for index in range(1000):
        old_split = 'train' if index < 800 else 'val' if index < 900 else 'test'
        real = ROOT / old_split / 'real' / f'real_{index:05d}.jpg'
        fake = ROOT / old_split / 'fake' / f'fake_{index:05d}.jpg'
        if not real.is_file() or not fake.is_file():
            missing += 1
            continue
        if not valid(real) or not valid(fake):
            corrupt += 1
            continue
        real_hash, fake_hash = sha(real), sha(fake)
        if real_hash in seen_real or fake_hash in seen_fake or real_hash == fake_hash:
            duplicates += 1
            continue
        seen_real.add(real_hash)
        seen_fake.add(fake_hash)
        pairs.append((index, real.resolve(), fake.resolve()))

    if len(pairs) < 30:
        raise SystemExit(f'Only {len(pairs)} valid matched pairs; need at least 30. No manifest written.')
    random.Random(SEED).shuffle(pairs)
    n = len(pairs)
    n_train = int(n * .8)
    n_val = int(n * .1)
    rows = []
    counts = {'train': 0, 'val': 0, 'test': 0}
    for pos, (index, real, fake) in enumerate(pairs):
        split = 'train' if pos < n_train else 'val' if pos < n_train + n_val else 'test'
        counts[split] += 1
        # BOTH members of the same original source row share one group.
        for path, label in ((real, 'real'), (fake, 'fake')):
            rows.append({'path': str(path), 'label': label, 'split': split,
                         'source': 'ELSA_D3_image_gen0_pilot', 'group': f'elsa_row_{index:05d}'})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'split', 'source', 'group'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Matched pairs: {n} | Missing one member: {missing} | Corrupt: {corrupt} | Exact duplicates: {duplicates}')
    print('Split counts (each pair has 1 real + 1 fake):')
    for split, count in counts.items():
        print(f'  {split}: {count} real + {count} fake')
    print(f'Manifest created: {OUT}')
    print('WARNING: Same-source near duplicates and real-image source validity need further audit before publication.')


if __name__ == '__main__':
    main()
