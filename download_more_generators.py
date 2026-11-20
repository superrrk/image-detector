"""Download additional ELSA_D3 generators for EXISTING TRAINING ROWS ONLY.

Run from the image-forensics-lab project root. Does not modify old data,
validation, test, or the current trained model. Downloaded samples require a
new training manifest before they have any effect on model accuracy.
"""
import csv
import hashlib
from io import BytesIO
from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageOps

ROOT = Path('dataset')
DEST = ROOT / 'extra_generators' / 'train' / 'fake'
DEST.mkdir(parents=True, exist_ok=True)
META = ROOT / 'extra_generators' / 'train_generator_metadata.csv'
MAX_ROWS = 800

# Only include rows with an already downloaded matched original training pair.
# File identifiers in your first downloader were sequential saved-row numbers.
rows = load_dataset('elsaEU/ELSA_D3', split='train', streaming=True)
existing = []
if META.exists():
    with META.open(newline='') as f:
        existing = list(csv.DictReader(f))
seen = {r['file'] for r in existing}
added = 0
errors = 0
try:
    for i, row in enumerate(rows):
        if i >= MAX_ROWS:
            break
        original_real = ROOT / 'train' / 'real' / f'real_{i:05d}.jpg'
        original_fake = ROOT / 'train' / 'fake' / f'fake_{i:05d}.jpg'
        if not (original_real.is_file() and original_fake.is_file()):
            continue
        # The first downloader named images by saved index, not necessarily
        # ELSA row number. Verify the original gen0 JPEG bytes before pairing.
        gen0 = row.get('image_gen0')
        if not isinstance(gen0, Image.Image):
            continue
        check = BytesIO()
        gen0.convert('RGB').save(check, 'JPEG', quality=95)
        if hashlib.sha256(check.getvalue()).digest() != hashlib.sha256(original_fake.read_bytes()).digest():
            print(f'Skipping unverified source row {i}', flush=True)
            continue
        for g in (1, 2, 3):
            img = row.get(f'image_gen{g}')
            if not isinstance(img, Image.Image):
                continue
            name = f'row_{i:05d}_gen{g}.jpg'
            target = DEST / name
            if target.exists():
                continue
            try:
                # Consistent output encoding across generators; avoids preserving
                # generator-specific source file formats as an easy shortcut.
                img = ImageOps.exif_transpose(img).convert('RGB')
                img.save(target, 'JPEG', quality=92, optimize=True)
                existing.append({
                    'file': str(target), 'source_row': str(i),
                    'source_id': str(row.get('id', '')),
                    'generator': str(row.get(f'model_gen{g}', f'gen{g}')),
                    'split': 'train', 'label': 'fake',
                })
                added += 1
                if added % 100 == 0:
                    print(f'Saved {added} additional train images', flush=True)
            except (OSError, ValueError) as e:
                errors += 1
                print(f'Failed row {i} gen{g}: {e}', flush=True)
finally:
    META.parent.mkdir(parents=True, exist_ok=True)
    with META.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file','source_row','source_id','generator','split','label'])
        writer.writeheader()
        writer.writerows(existing)

print(f'Finished. New images: {added}; total tracked: {len(existing)}; failures: {errors}')
print('Saved to:', DEST)
print('Metadata:', META)
print('IMPORTANT: these images are not yet included in paired_manifest.csv.')
