import torch
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import EyeDataset
from src.augmentations import train_transforms # или train_transforms, если нужно
from src.model import EyeClassifierCNN  

def visualize_predictions(model, dataset, device='cpu', num_samples=8):
    model.eval()
    model.to(device)

    indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].item()

            output = model(image)
            pred = torch.argmax(output, dim=1).item()

            images.append(sample["image"])
            true_labels.append(label)
            pred_labels.append(pred)

    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2.5, 3))
    for i, ax in enumerate(axs):
        img = images[i].squeeze().numpy()
        true_class = dataset.classes[true_labels[i]]
        pred_class = dataset.classes[pred_labels[i]]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"T: {true_class}\nP: {pred_class}", fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Настрой путь к данным и модели
    dataset = EyeDataset(
        root_dir="data",
        image_size=(224, 224),
        transform=train_transforms
    )

    model = EyeClassifierCNN(num_classes=4) 
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location='cpu'))

    visualize_predictions(model, dataset, device='cpu', num_samples=6)
