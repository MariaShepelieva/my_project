import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.MotionBlur(p=0.3),
    A.MedianBlur(blur_limit=3, p=0.2),
    A.OpticalDistortion(distort_limit=0.05, p=0.3),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])
