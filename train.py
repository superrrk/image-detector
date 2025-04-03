import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = 'data'
BATCH_SIZE = 32
EPOCHS = 5
IMAGE_SIZE = 128
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/fake_detector.pth'

# Transforms 
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Model
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

model = FakeImageCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training 
def train():
    print("📂 Loading training dataset...")
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform)
    print(f"🧮 Training samples: {len(train_dataset)}")

    print("📂 Loading validation dataset...")
    val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=transform)
    print(f"🧪 Validation samples: {len(val_dataset)}")

    print("🚚 Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"📉 Training loss: {running_loss:.4f}")
        validate(val_loader)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# Validation
def validate(val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"✅ Validation accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
