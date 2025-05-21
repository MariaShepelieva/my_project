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
    assert len(dataset) > 0, "Error: Dataset is empty!"


def test_image_shape(dataset):
    image, label, label_ohe = dataset[0]["image"], dataset[0]["label"], dataset[0]["label_ohe"]
    assert isinstance(image, torch.Tensor), "Error: Image is not a torch.Tensor!"
    assert image.shape == (1, 224, 224), f"Error: Expected shape (1, 224, 224), but got {image.shape}!"
    assert label_ohe.shape == (len(dataset.classes),), f"Error: Expected shape ({len(dataset.classes)},), but got {label_ohe.shape}!"
    assert label.dtype == torch.long, "Error: Label must be of type torch.long!"


def test_label_type(dataset):
    label = dataset[0]["label"]
    assert isinstance(label, torch.Tensor), "Error: Label is not a torch.Tensor!"
    assert label.dtype == torch.long, "Error: Label must be of type torch.long!"
    assert 0 <= label < len(dataset.classes), f"Error: Label {label} is out of bounds for classes!"


def test_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    images, labels = batch["image"], batch["label"]

    assert images.shape == (16, 1, 224, 224), f"Error: incorrect batch shape {images.shape}!"
    assert labels.shape == (16,), f"Error: incorrect label shape {labels.shape}!"


def test_multiple_images(dataset):
    indices = [0, 5, 10]
    for idx in indices:
        image = dataset[idx]["image"]
        assert image.shape == (1, 224, 224), f"Error: Incorrect image shape for index {idx}!"
