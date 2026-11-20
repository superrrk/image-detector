"""Fine-tune ResNet18, tune on validation, report on untouched test split.

Usage: python -m ml.train --manifest dataset/manifest.csv --epochs 8 --pretrained
Set --holdout-source to enforce zero-shot generator assessment on test.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from ml.data import ImageManifestDataset, read_manifest
from ml.metrics import evaluate, find_threshold, source_report
from ml.model import build_model, transform


def predict_loader(model, loader, device):
    model.eval()
    truth, scores, sources = [], [], []
    with torch.inference_mode():
        for images, labels, origins in loader:
            logits = model(images.to(device)).flatten()
            truth.extend(labels.int().tolist())
            scores.extend(logits.sigmoid().cpu().tolist())
            sources.extend(origins)
    return truth, scores, sources


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_manifest(args.manifest, verify_hashes=args.verify_hashes)
    if args.holdout_source:
        source = args.holdout_source
        if any(r['source'] == source and r['split'] != 'test' for r in rows):
            raise ValueError(f'Holdout source {source!r} must appear only in test')
        if not any(r['source'] == source and r['split'] == 'test' for r in rows):
            raise ValueError(f'No test samples for holdout source {source!r}')
    datasets = {split: ImageManifestDataset(rows, split, transform(split == 'train')) for split in ('train', 'val', 'test')}
    labels = np.array([int(r['label'] == 'fake') for r in datasets['train'].rows])
    counts = np.bincount(labels, minlength=2)
    if min(counts) == 0:
        raise ValueError('Both classes required in training')
    weights = 1 / counts[labels]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, sampler=WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator), num_workers=args.workers)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(pretrained=args.pretrained, freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device)).flatten()
            loss = loss_fn(logits, targets.to(device).float())
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(targets)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, targets, _ in val_loader:
                loss = loss_fn(model(images.to(device)).flatten(), targets.to(device).float())
                val_loss += float(loss.item()) * len(targets)
        val_loss /= len(datasets['val'])
        print(f'epoch {epoch+1}/{args.epochs}: train_loss={train_loss/len(datasets["train"]):.4f} val_loss={val_loss:.4f}', flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'architecture': 'resnet18-v2', 'model_state_dict': model.cpu().state_dict(), 'image_size': 224,
                        'pretrained_backbone': args.pretrained, 'epoch': epoch+1, 'seed': args.seed,
                        'class_to_index': {'real': 0, 'fake': 1}}, output)
            model.to(device)
    # IMPORTANT: reload the best epoch before validation threshold selection.
    checkpoint = torch.load(output, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    val_y, val_scores, _ = predict_loader(model, val_loader, device)
    threshold = find_threshold(val_y, val_scores, max_fpr=args.target_fpr)
    checkpoint['threshold'] = threshold
    checkpoint['validation_metrics'] = evaluate(val_y, val_scores, threshold)
    torch.save(checkpoint, output)
    test_y, test_scores, test_sources = predict_loader(model, test_loader, device)
    report = {
        'model': 'resnet18-v2', 'threshold_selected_on': 'validation',
        'max_validation_false_positive_rate_target': args.target_fpr,
        'split_counts': {name: len(ds) for name, ds in datasets.items()},
        'validation': checkpoint['validation_metrics'],
        'test': evaluate(test_y, test_scores, threshold),
        'test_by_source': source_report(test_y, test_scores, test_sources, threshold),
        'limitations': 'Do not interpret raw scores as calibrated probabilities. Independent generators and compression shifts must be evaluated.'
    }
    report_path = output.with_suffix('.metrics.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Checkpoint: {output} | Metrics: {report_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', default='models/image_origin_resnet18_v2.pth')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--target-fpr', type=float, default=0.05)
    p.add_argument('--pretrained', action='store_true', help='Download ImageNet backbone weights if not cached')
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--verify-hashes', action='store_true')
    p.add_argument('--holdout-source', default='')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.target_fpr <= 1:
        p.error('epochs and batch-size must be positive; target-fpr must be in [0,1]')
    return args


if __name__ == '__main__':
    run(parse_args())
