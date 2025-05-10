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

from src.work import EyeDataset
from src.augmentations import train_transforms



def check_label_folder_alignment(dataset):
    print("Проверка соответствия меток и папок...")
    for path, label in zip(dataset.image_paths, dataset.labels):
        folder_name = path.parent.name
        expected_label = dataset.classes.index(folder_name)
        assert label == expected_label, f"[ОШИБКА] {path}: ожидалась метка {expected_label}, получена {label}"
    print("Все метки соответствуют папкам.\n")


def check_class_distribution(dataset):
    print("Распределение классов:")
    counts = Counter(dataset.labels)
    for idx, count in counts.items():
        class_name = dataset.classes[idx]
        print(f" - {class_name} ({idx}): {count} изображений")
    print()


def check_images_readable(dataset):
    print("Проверка целостности изображений...")
    for path in dataset.image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[ОШИБКА] Не удалось прочитать изображение: {path}")
    print("Проверка чтения изображений завершена.\n")


def preview_random_images(dataset):
    indices = np.random.choice(len(dataset), size=5, replace=False)
    for i in indices:
        image, label = dataset[i]
        
        image = image.permute(1, 2, 0)
        image = image * 0.5 + 0.5

        plt.imshow(image.numpy())
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.savefig(f'random_image_{i}.png') 
        plt.close()




if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "src" / "EyeDataset"
    dataset = EyeDataset(root_dir=dataset_path, image_size=(224, 224), transform=train_transforms)

    print(f"Проверка набора данных: {dataset_path}")
    print(f"Кол-во изображений: {len(dataset)}\n")

    check_label_folder_alignment(dataset)
    check_class_distribution(dataset)
    check_images_readable(dataset)
    preview_random_images(dataset)
