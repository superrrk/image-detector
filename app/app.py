from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency
from flask_scss import Scss

app = Flask(__name__)
Scss(app, static_dir='static', asset_dir='static/styles')
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model Setup
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

model = FakeImageCNN()
model.load_state_dict(torch.load('models/fake_detector.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    img_tensor.requires_grad = True

    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()
        is_fake = prob > 0.5
        label = "FAKE" if is_fake else "REAL"
        confidence = prob if is_fake else 1 - prob

    # Saliency Map 
    saliency = Saliency(model)
    grads = saliency.attribute(img_tensor, target=0)
    grads = grads.squeeze().detach().numpy()
    grayscale = np.maximum(grads, 0).mean(axis=0)
    grayscale /= grayscale.max()

    # Save saliency map
    plt.figure(figsize=(5, 5))
    plt.imshow(img.resize((128, 128)))
    plt.imshow(grayscale, cmap='hot', alpha=0.5)
    plt.axis('off')
    plt.title(f"{label} ({confidence*100:.2f}%)")
    plt.tight_layout()
    plt.savefig("static/saliency_map.png")
    plt.close()

    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)
            return render_template('index.html',
                                   filename=filename,
                                   label=label,
                                   confidence=f"{confidence*100:.2f}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
