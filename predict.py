import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import Saliency

# === Config ===
MODEL_PATH = 'models/fake_detector.pth'
IMAGE_PATH = sys.argv[1]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# === CNN Class ===
class FakeImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# === Load Model ===
model = FakeImageCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Load Image ===
img = Image.open("client/public/samples/real_ed.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)
img_tensor.requires_grad = True

# === Prediction ===
with torch.no_grad():
    output = model(img_tensor)
    prob = output.item()
    is_fake = prob > 0.5

label = "FAKE" if is_fake else "REAL"
confidence = prob if is_fake else 1 - prob
print(f"🧠 Prediction: {label} ({confidence*100:.2f}% confidence)")

# === Saliency Map ===
saliency = Saliency(model)
grads = saliency.attribute(img_tensor, target=0)
grads = grads.squeeze().cpu().numpy()

# === Normalize and Visualize ===
grayscale = np.maximum(grads, 0).mean(axis=0)
grayscale /= grayscale.max()

plt.figure(figsize=(6, 6))
plt.imshow(img.resize((128, 128)))
plt.imshow(grayscale, cmap='hot', alpha=0.5)
plt.axis('off')
plt.title(f"Saliency Map - {label}")
plt.savefig("saliency_map.png")
print("🧠 Saved saliency map to saliency_map.png")


