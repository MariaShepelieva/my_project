import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# шлях до батьківської директорії
sys.path.append(str(Path(__file__).resolve().parent.parent))
<<<<<<< HEAD
from src.work import EyeDataset
=======
from work import EyeDataset
>>>>>>> 0721a30 (PathLib and translate outputs to ukrainian)


@pytest.fixture(scope="module")
def dataset():
<<<<<<< HEAD
    dataset_path = Path(__file__).parent.parent / "src" / "EyeDataset"
=======
    dataset_path = Path(r"C:/Coursework/EyeDataset")
>>>>>>> 0721a30 (PathLib and translate outputs to ukrainian)
    return EyeDataset(root_dir=dataset_path, image_size=(224, 224))


def test_dataset_length(dataset):
    assert len(dataset) > 0, "Помилка: Датасет порожній!"


def test_image_shape(dataset):
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor), "Помилка: Зображення не в форматі torch.Tensor!"
    assert image.shape == (3, 224, 224), f"Помилка: Очікується форма (3, 224, 224), але отримали {image.shape}!"


def test_label_type(dataset):
    _, label = dataset[0]
    assert isinstance(label, torch.Tensor), "Помилка: Мітка не в форматі torch.Tensor!"
    assert label.dtype == torch.long, "Помилка: Мітка має бути типу torch.long!"
    assert 0 <= label < len(dataset.classes), f"Помилка: Метка {label} виходить за діапазон класів!"


def test_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    images, labels = batch

    assert images.shape == (16, 3, 224, 224), f"Помилка: неправильна форма batch {images.shape}!"
    assert labels.shape == (16,), f"Помилка: неправильна форма міток {labels.shape}!"


def test_multiple_images(dataset):
    indices = [0, 5, 10] 
    images = [dataset[i][0] for i in indices]

    for img in images:
        assert img.shape == (3, 224, 224), f"Помилка: Неправильна форма зображення {img.shape}!"
