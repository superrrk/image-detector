# Image-origin classifier v2 — training and evidence

## What changed

1. **Transfer learning backbone:** ResNet18 is initialized from ImageNet weights with `--pretrained` instead of optimizing a large randomly initialized dense head from scratch. ResNet18 is a pragmatic CPU-friendly baseline, *not* a proven best detector.
2. **Stable loss:** single logit and `BCEWithLogitsLoss` rather than sigmoid + BCE; explicit mapping `real=0`, `fake=1`.
3. **Artifact robustness:** random JPEG encoding, color jitter, crop, and flip during training to reduce accidental file-format and compression shortcuts. This is an ablation hypothesis, not a measured improvement.
4. **Anti-leakage manifest:** CSV with `path,label,split,source,group`; reject repeated filenames or group identifiers spanning partitions, optionally SHA-256 byte duplicates. Group identifiers must represent the *same subject or original image* where known. Automatically generated group IDs do not prove identity separation.
5. **Validation-only decisions:** save best epoch by validation loss, select threshold against a validation false-positive-rate target. Evaluate once on the untouched test split.
6. **Per-generator analysis:** save precision, recall, F1, ROC-AUC, average precision, confusion matrix and false-positive rate for test images broken down by `source`.
7. **Application integration:** Predictor detects `resnet18-v2` checkpoints and applies its own 224px ImageNet normalization. Old checkpoints retain legacy inference.

## Install

```bash
pip install -r requirements-train.txt
```

## Prepare data

For a dataset organized as `dataset/train/real`, `dataset/train/fake`, `dataset/val/real`, etc.:

```bash
python -m ml.prepare_manifest --root dataset --output dataset/manifest.csv
```

Edit the manifest's `source` column to record actual generation models / origins and `group` to represent shared subjects or base images across variants. `unknown` is **not** a generator-specific benchmark. The generated manifest cannot automatically detect near duplicates or identity leakage. Use a license-compliant dataset; do not mix conflicting original-label conventions.

Your historical GitHub repository committed images under `real_vs_fake/real-vs-fake`. Verify the actual available split folders and label meaning before using them.

## Train and evaluate

```bash
python -m ml.train \
  --manifest dataset/manifest.csv \
  --pretrained --epochs 8 --batch-size 32 \
  --target-fpr 0.05 --verify-hashes \
  --output models/image_origin_resnet18_v2.pth
```

`--pretrained` downloads ImageNet weights if they are not cached and requires network access. To test the pipeline without a download, omit it; that will train from scratch and should not be presented as the intended stronger baseline. Model training may take substantial time and GPU resources.

The checkpoint and `models/image_origin_resnet18_v2.metrics.json` are written locally. The JSON holds real evaluation metrics after training (not projected numbers). To test across unseen generators, reserve a named source exclusively for test and pass `--holdout-source <name>`; consider full leave-one-generator-out experiments.

Set `MODEL_PATH=models/image_origin_resnet18_v2.pth` when launching the web app. No compatible checkpoint is bundled in this package.

## Scientific caveats

- Old reported **87% validation accuracy** is not reproduced here; no improved accuracy result is claimed.
- Fake-image detection cannot reliably establish provenance; metadata, ELA and model gradients are not proof.
- ImageNet pretrained features, identity leakage and JPEG shortcuts can all artificially inflate metrics. Use cross-generator and post-processing stress tests, plus multiple random seeds.
- A raw sigmoid score is **not calibrated confidence**. Further temperature scaling needs an independent calibration partition before displaying probabilities.
- For a portfolio showcase, report the exact dataset version, source counts, splits, hardware, training time and uncertainty intervals; avoid cherry-picking a convenient threshold.

## Research context

- Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image* (NeurIPS 2023): cross-generator tests matter. https://papers.nips.cc/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html
- Li et al., *Bridging the Gap Between Ideal and Real-world Evaluation* (ICCV 2025): investigate distribution shifts. https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.pdf
- *GAMMA: Generalizable Alignment via Multi-task and Manipulation-Augmented Training* (2025): manipulation robustness is an active research direction. https://arxiv.org/abs/2509.10250
