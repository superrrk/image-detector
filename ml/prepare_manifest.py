"""Build manifest from existing train/val/test/{real,fake} folder layout.

Source labels should be replaced with actual generator/camera provenance before
claiming generator holdout results. A filename is NOT proof of distinct identity.
"""
import argparse
import csv
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Dataset root containing train/val/test subdirectories')
    parser.add_argument('--output', default='dataset/manifest.csv')
    parser.add_argument('--source', default='unknown', help='Source identifier when dataset lacks provenance')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for split in ('train', 'val', 'test'):
        for label in ('real', 'fake'):
            for image in sorted((root / split / label).glob('*')):
                if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
                    continue
                # Each file gets a distinct group as a default; inspect near-duplicate
                # and same-identity leakage manually before presenting test metrics.
                group = hashlib.sha256(str(image).encode()).hexdigest()[:16]
                data.append({'path': str(image), 'label': label, 'split': split,
                             'source': args.source, 'group': group})
    if not data:
        parser.error('No images found; expected root/train/real etc.')
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'label', 'split', 'source', 'group'])
        writer.writeheader()
        writer.writerows(data)
    print(f'Wrote {len(data)} samples to {output}. WARNING: Review source and group fields for leakage.')


if __name__ == '__main__':
    main()
