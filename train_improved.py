import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from PIL import Image
import random

DATA_DIR = 'data'
BATCH_SIZE = 32
EPOCHS = 10  # Increased epochs
IMAGE_SIZE = 128
LR = 1e-4  # Reduced learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/fake_detector_improved.pth'

# Enhanced transforms for better generalization
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FakeImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

def find_best_threshold(model, val_loader):
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            # Convert to binary format explicitly
            all_outputs.extend(outputs.cpu().numpy().flatten())
            # Convert integer labels to binary format
            binary_labels = labels.float().numpy()
            all_labels.extend(binary_labels)
    
    # Convert lists to numpy arrays
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find the threshold that gives the best F1 score
    if len(thresholds) > 0:
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    else:
        best_threshold = 0.5
    
    return best_threshold

def train():
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=transform_val)
    
    # Calculate class weights for balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / torch.Tensor(class_counts)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
    
    # Initialize model and training
    model = FakeImageCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0
    best_threshold = 0.5
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        print(f"📉 Training - Loss: {running_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
                outputs = model(images)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"✅ Validation accuracy: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_threshold = find_best_threshold(model, val_loader)
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold,
                'val_accuracy': best_val_acc
            }, MODEL_PATH)
            print(f"✨ New best model saved! Threshold: {best_threshold:.4f}")

if __name__ == "__main__":
    train() 