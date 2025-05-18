import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os
import rootutils


rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from src.dataset import EyeDataset


@pytest.fixture(scope="module")
def dataset():
    dataset_path = Path(os.path.abspath(__file__)).parent.parent / "data" 
    return EyeDataset(root_dir=dataset_path, image_size=(224, 224))


def test_dataset_length(dataset):
    assert len(dataset) > 0, "Помилка: Датасет порожній!"


def test_image_shape(dataset):
    image, label, label_ohe = dataset[0]["image"], dataset[0]["label"], dataset[0]["label_ohe"]
    assert isinstance(image, torch.Tensor), "Помилка: Зображення не в форматі torch.Tensor!"
    assert image.shape == (1, 224, 224), f"Помилка: Очікується форма (1, 224, 224), але отримали {image.shape}!"
    assert label_ohe.shape == (len(dataset.classes),), f"Помилка: Очікується форма ({len(dataset.classes)},), але отримали {label_ohe.shape}!"
    assert label.dtype == torch.long, "Помилка: Мітка має бути типу torch.long!"


def test_label_type(dataset):
    label = dataset[0]["label"]
    assert isinstance(label, torch.Tensor), "Помилка: Мітка не в форматі torch.Tensor!"
    assert label.dtype == torch.long, "Помилка: Мітка має бути типу torch.long!"
    assert 0 <= label < len(dataset.classes), f"Помилка: Мітка {label} виходить за діапазон класів!"


def test_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    images, labels = batch["image"], batch["label"]

    assert images.shape == (16, 1, 224, 224), f"Помилка: неправильна форма batch {images.shape}!"
    assert labels.shape == (16,), f"Помилка: неправильна форма міток {labels.shape}!"


def test_multiple_images(dataset):
    indices = [0, 5, 10]
    for idx in indices:
        image = dataset[idx]["image"]
        assert image.shape == (1, 224, 224), f"Помилка: Неправильна форма зображення для індексу {idx}!"
