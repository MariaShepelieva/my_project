import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=3)
    ], p=0.3),
    A.OpticalDistortion(distort_limit=0.05, p=0.3),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(p=0.3)
])

valid_transforms = A.Compose([])
