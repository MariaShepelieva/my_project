from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from functools import lru_cache
import hashlib

class EyeDataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), transform=None):
        """
        :param root_dir: Папка с изображениями, в которой подпапки = классы.
        :param image_size: Кортеж (H, W) – во что ресайзить изображения.
        :param transform: Дополнительные аугментации (из torchvision.transforms или Albumentations).
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        seen_hashes = set()
        for label, class_name in enumerate(self.classes):
            class_path = self.root_dir / class_name
            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue

                img_hash = self._get_image_hash(str(img_path))
                if not img_hash or img_hash in seen_hashes:
                    # Пустой хеш или дубликат — пропускаем
                    continue

                seen_hashes.add(img_hash)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        # Загрузка и ресайз в grayscale
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size)
        image = image[:, :, None]  # H × W → H × W × 1

        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = image.astype('float32') / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # (1, H, W)

        label_tensor = torch.tensor(label, dtype=torch.long)
        label_ohe    = torch.zeros(len(self.classes), dtype=torch.float32)
        label_ohe[label] = 1.0

        return {
            "image": image,
            "label": label_tensor,
            "label_ohe": label_ohe
        }

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_image_hash(path: str) -> str:

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return ""
        return hashlib.md5(img.tobytes()).hexdigest()


if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    dataset_path = Path(os.path.abspath(__file__)).parent.parent / "data"
    dataset = EyeDataset(root_dir=dataset_path, image_size=(224, 224))

    print(f"Number of images in the set: {len(dataset)}")
    image, label, label_ohe = dataset[0]["image"], dataset[0]["label"], dataset[0]["label_ohe"]
    print(f"Image shape: {image.shape} (Expected: [1, 224, 224]), Class: {label}")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    images, labels, labels_ohe = batch["image"], batch["label"], batch["label_ohe"]
    print(f"Batch shape: {images.shape} (Expected: [16, 1, 224, 224]), Classes: {labels}")
    print(f"Classes: {dataset.classes}")
