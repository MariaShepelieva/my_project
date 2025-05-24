import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.dataset import EyeDataset
from src.model import EyeClassifierCNN
from src.augmentations import train_transforms

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
VAL_SPLIT = 0.2  # 20% for validation

# Data preparation
data_path = Path("data")
dataset = EyeDataset(data_path, image_size=(224, 224), transform=train_transforms)
num_classes = len(dataset.classes)  # number of classes

total_size = len(dataset)
val_size = int(total_size * VAL_SPLIT)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeClassifierCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()

    avg_train_loss = train_loss / train_size
    train_acc = train_correct / train_size

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / val_size
    val_acc = val_correct / val_size

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # Checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = checkpoint_dir / 'best_model.pth'
        torch.save(model.state_dict(), save_path)

print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
