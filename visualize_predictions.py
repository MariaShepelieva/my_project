import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import EyeDataset
from src.augmentations import train_transforms
from src.model import EyeClassifierCNN


def visualize_predictions(model, dataset, device='cpu', num_samples=6, save_path='predictions.png'):
    model.eval()
    model.to(device)

    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2.5, 3))

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            sample = dataset[idx]
            img = sample['image'].cpu().squeeze().numpy()
            true_label = sample['label'].item()

            # Inference
            img_tensor = torch.tensor(sample['image']).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred_label = torch.argmax(output, dim=1).item()

            # Denormalize and display
            img_disp = ((img * 0.5) + 0.5) * 255
            img_disp = np.clip(img_disp, 0, 255).astype(np.uint8)

            ax.imshow(img_disp, cmap='gray')
            ax.set_title(f"T: {dataset.classes[true_label]}\nP: {dataset.classes[pred_label]}", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Predictions visualization saved to: {save_path}")


if __name__ == '__main__':
    # Load dataset and model
    dataset = EyeDataset(root_dir='data', image_size=(224, 224), transform=train_transforms)
    model = EyeClassifierCNN(num_classes=len(dataset.classes))
    checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint)

    # Generate and save visualization
    visualize_predictions(model, dataset, device='cpu', num_samples=6, save_path='predictions.png')