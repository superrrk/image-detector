import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

# Constants
DATA_DIR = 'data'
BATCH_SIZE = 32
EPOCHS = 20
IMAGE_SIZE = 224  # Standard size for many pre-trained models
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/best_model.pth'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def create_model():
    # Use a pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model

def train():
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Prepare data
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'train'),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'val'),
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_model().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(DEVICE)
            # Ensure labels are binary and properly shaped
            labels = labels.float().to(DEVICE).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        print(f"📉 Training - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                # Ensure labels are binary and properly shaped
                labels = labels.float().to(DEVICE).view(-1, 1)
                outputs = model(images)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"✅ Validation accuracy: {val_acc:.4f}")
        
        # Find best threshold using validation set
        best_threshold = find_best_threshold(model, val_loader)
        print(f"📊 Best threshold: {best_threshold:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold,
                'val_accuracy': val_acc
            }, MODEL_PATH)
            print(f"✨ New best model saved!")

def find_best_threshold(model, val_loader):
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            # Convert labels to binary format (0 or 1)
            labels = labels.float().to(DEVICE)
            outputs = model(images)
            # Ensure outputs and labels are the right shape
            outputs = outputs.squeeze().cpu().numpy()
            labels = labels.cpu().numpy()
            all_outputs.extend(outputs)
            all_labels.extend(labels)
    
    # Convert to numpy arrays and ensure binary format
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels, dtype=np.int32)
    
    # Print shapes and unique values for debugging
    print(f"Output shape: {all_outputs.shape}, Label shape: {all_labels.shape}")
    print(f"Unique labels: {np.unique(all_labels)}")
    print(f"Output range: [{all_outputs.min():.3f}, {all_outputs.max():.3f}]")
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find the threshold that gives the best F1 score
    if len(thresholds) > 0:
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    else:
        best_threshold = 0.5
    
    return best_threshold

if __name__ == "__main__":
    train() 