from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from src.augmentations import train_transforms


class EyeDataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), transform=None):
        """
        :param root_dir: Папка с изображениями, в которой подпапки = классы.
        :param image_size: Кортеж (H, W) – во что ресайзить изображения.
        :param transform: Дополнительные аугментации (из torchvision.transforms).
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        for label, class_name in enumerate(self.classes):
            class_path = self.root_dir / class_name
            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:

            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, torch.tensor(label, dtype=torch.long)



if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "EyeDataset"
    dataset = EyeDataset(root_dir=dataset_path, image_size=(224, 224))
   
    print(f"Кількість зображень в наборі: {len(dataset)}")
   
    image, label = dataset[0]
    print(f"Форма зображення: {image.shape} (Очікується: [3, 224, 224]), Клас: {label}")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    batch = next(iter(dataloader))
    images, labels = batch
    print(f"Форма batch: {images.shape} (Очікується: [16, 3, 224, 224]), Классы: {labels}")
    print(dataset.classes)