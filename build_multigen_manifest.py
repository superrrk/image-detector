"""Merge verified additional ELSA_D3 generator outputs into the TRAIN split only.

Run from the image-forensics-lab project root:
    python build_multigen_manifest.py

Inputs:
    dataset/paired_manifest.csv
    dataset/extra_generators/train_generator_metadata.csv
Output:
    dataset/multigen_manifest.csv

Does not modify source images, original manifest, model, validation or test rows.
"""
from __future__ import annotations
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path('dataset')
BASE = ROOT / 'paired_manifest.csv'
META = ROOT / 'extra_generators' / 'train_generator_metadata.csv'
DEST = ROOT / 'multigen_manifest.csv'


def main() -> None:
    with BASE.open(newline='', encoding='utf-8') as f:
        base = list(csv.DictReader(f))
    if not base:
        raise SystemExit('Empty paired_manifest.csv. Run prepare_pairs.py first.')
    needed = {'path', 'label', 'split', 'source', 'group'}
    if not needed.issubset(base[0]):
        raise SystemExit('Your paired_manifest.csv lacks expected columns.')
    splits_by_group = {}
    for row in base:
        group, split = row['group'], row['split']
        if group in splits_by_group and splits_by_group[group] != split:
            raise SystemExit(f'Existing manifest leaks group across splits: {group}')
        splits_by_group[group] = split
    if not META.is_file():
        raise SystemExit('Missing generator metadata. Run download_more_generators.py first.')
    with META.open(newline='', encoding='utf-8') as f:
        extra = list(csv.DictReader(f))
    combined = list(base)
    seen_paths = {str(Path(r['path']).resolve()) for r in base}
    eligible = 0
    skipped_nontrain = 0
    missing = 0
    repeated = 0
    unrecognized = 0
    for row in extra:
        try:
            source_row = int(row['source_row'])
        except (KeyError, ValueError):
            unrecognized += 1
            continue
        group = f'elsa_row_{source_row:05d}'
        split = splits_by_group.get(group)
        if split != 'train':
            skipped_nontrain += 1
            continue
        raw = Path(row['file'])
        path = raw if raw.is_file() else Path.cwd() / raw
        if not path.is_file():
            missing += 1
            continue
        resolved = str(path.resolve())
        if resolved in seen_paths:
            repeated += 1
            continue
        # Only accept script-generated image file names and matching source rows.
        if not re.fullmatch(rf'row_{source_row:05d}_gen[123]\.jpg', path.name):
            unrecognized += 1
            continue
        seen_paths.add(resolved)
        combined.append({'path': resolved, 'label': 'fake', 'split': 'train',
                         'source': f'ELSA_D3_{row.get("generator") or path.stem.split("_")[-1]}',
                         'group': group})
        eligible += 1
    if eligible == 0:
        raise SystemExit('No eligible extra images for the clean training split; no manifest written.')
    # Preserve existing validation/test rows exactly; no extra generator outputs in either.
    assert all(r['split'] == 'train' for r in combined[len(base):])
    with DEST.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'split', 'source', 'group'])
        writer.writeheader()
        writer.writerows(combined)
    for split in ('train', 'val', 'test'):
        c = Counter(r['label'] for r in combined if r['split'] == split)
        print(f'{split}: {c["real"]} real, {c["fake"]} fake')
    print(f'Added {eligible} verified training images; skipped {skipped_nontrain} held-out, {missing} missing, {repeated} duplicates, {unrecognized} invalid entries.')
    print(f'Wrote {DEST} (original manifest and model unchanged)')
    print('Note: This is a pilot, not independent cross-generator evaluation.')


if __name__ == '__main__':
    main()
