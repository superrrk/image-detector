"""ResNet18 image-origin baseline shared by training and inference.

A frozen or fine-tuned pretrained backbone is a stronger starting point than
training a tiny CNN from scratch, but benchmark gains must be measured.
"""
from __future__ import annotations
import io
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class RandomJPEG:
    """Prevent the model from exploiting PNG-vs-JPEG dataset shortcuts."""
    def __init__(self, probability=0.5, quality_range=(55, 95)):
        self.probability = probability
        self.quality_range = quality_range

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=random.randint(*self.quality_range))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def transform(training=False):
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15)], p=.6),
            RandomJPEG(probability=.65),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def build_model(pretrained=False, freeze_backbone=False):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, 1)  # logits, no sigmoid: stable BCEWithLogitsLoss
    return model


def predict_scores(model, image, device="cpu", multi_crop=False):
    """Return AI-like score, optionally averaging logits across five crops.

    The score is NOT necessarily calibrated; choose thresholds on validation only.
    """
    t = transform(False)
    model.eval()
    if not multi_crop:
        with torch.inference_mode():
            logit = model(t(image).unsqueeze(0).to(device)).flatten().mean()
        return float(logit.sigmoid().item())
    from torchvision.transforms import FiveCrop
    # Use a scaled 256px square so all five crops cover different image regions.
    resized = ImageOps.fit(image, (256, 256), method=Image.Resampling.BICUBIC)
    crops = FiveCrop(IMAGE_SIZE)(resized)
    batch = torch.stack([transforms.Normalize(MEAN, STD)(transforms.ToTensor()(crop)) for crop in crops]).to(device)
    with torch.inference_mode():
        logits = model(batch).flatten()
    return float(logits.mean().sigmoid().item())
