import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2

class Normalize(ImageOnlyTransform):
    def apply(self, img, **params):
        return img/255.0

def get_tfms():
    return A.Compose([
        Normalize(always_apply=True),
        A.Resize(height=224, width=224, always_apply=True),
        ToTensorV2(always_apply=True)
    ])
