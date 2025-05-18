import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3)
    ], p=0.2),
    A.OpticalDistortion(distort_limit=0.02, p=0.2),
    A.GaussNoise(p=0.1),
    A.CoarseDropout(p=0.1),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])



# valid_transforms = A.Compose([])
