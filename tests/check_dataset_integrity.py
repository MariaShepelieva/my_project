import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import cv2
import torch
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

from src.dataset import EyeDataset
from src.augmentations import train_transforms



def check_label_folder_alignment(dataset):
    print("Checking Label and Folder Matching...")
    for path, label in zip(dataset.image_paths, dataset.labels):
        folder_name = path.parent.name
        expected_label = dataset.classes.index(folder_name)
        assert label == expected_label, f"[ERROR] {path}: expected label {expected_label}, received {label}"
    print("All labels correspond to folders.\n")


def check_class_distribution(dataset):
    print("Class distribution:")
    counts = Counter(dataset.labels)
    for idx, count in counts.items():
        class_name = dataset.classes[idx]
        print(f" - {class_name} ({idx}): {count} images")
    print()


def check_images_readable(dataset):
    print("Checking image readability...")
    for path in dataset.image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[ERROR] Could not read image: {path}")
    print("Image readability check completed.\n")


def preview_random_images(dataset):
    indices = np.random.choice(len(dataset), size=5, replace=False)
    for i in indices:
        sample = dataset[i]
        image_tensor = sample["image"]  # Tensor: shape (1, H, W)
        label = sample["label"]

        # Convert tensor to numpy and denormalize
        image = image_tensor.squeeze().numpy()  
        image = (image * 0.5 + 0.5) * 255 
        image = np.clip(image, 0, 255).astype(np.uint8)

    
        label_text = dataset.classes[label.item()]
        cv2.putText(image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

        cv2.imwrite(f"preview_aug_{i}.png", image)



if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "data"
    dataset = EyeDataset(root_dir=dataset_path, image_size=(224, 224), transform=train_transforms)

    print(f"Checking dataset: {dataset_path}")
    print(f"Number of images: {len(dataset)}\n")

    check_label_folder_alignment(dataset)
    check_class_distribution(dataset)
    check_images_readable(dataset)
    preview_random_images(dataset)
