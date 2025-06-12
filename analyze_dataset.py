# analyze_dataset.py
import cv2
import torch
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.dataset import EyeDataset
from src.model import EyeClassifierCNN
from src.augmentations import train_transforms

# Настройки
BATCH_SIZE = 16
SAVE_MISCL = True
os.makedirs("misclassified", exist_ok=True)

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeClassifierCNN(num_classes=len(EyeDataset("data", (224, 224)).classes))
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Загрузка датасета и DataLoader
dataset = EyeDataset("data", image_size=(224, 224), transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds = []
all_labels = []

# Инференс и сбор предсказаний
with torch.no_grad():
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Сохранение misclassified
        if SAVE_MISCL:
            for i in range(len(images)):
                if preds[i] != labels[i]:
                    img = images[i].cpu().squeeze().numpy() * 0.5 + 0.5
                    img = (img * 255).astype(np.uint8)
                    fname = f"misclassified/{len(all_preds)}_true_{dataset.classes[labels[i]]}_pred_{dataset.classes[preds[i]]}.png"
                    cv2.imwrite(fname, img)

# Classification report
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

# Confusion matrix
tt = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(tt, display_labels=dataset.classes).plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_static.png")
print("Confusion matrix saved as confusion_matrix_static.png")
