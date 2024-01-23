import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class AnimalsDataset(Dataset):

    def __init__(self, src, tfms):
        self.class_dict = self._get_dataset_dict(src=src)
        self.tfms = tfms

    def _get_dataset_dict(self, src):
        class_dict = dict()
        classes = os.listdir(src)
        self._encode_labels(classes=classes)
        for c in classes:
            filepaths = [os.path.join(src, f'{c}/{f}') for f in os.listdir(os.path.join(src,c))]
            clist = [c] * len(filepaths)
            class_dict.update(zip(filepaths, clist))
        return class_dict
    
    def _encode_labels(self, classes):
        self.encoding = {}
        c2int = torch.tensor([i for i,c in enumerate(classes)])
        one_hot = F.one_hot(c2int, num_classes=len(classes))
        for i, c in enumerate(classes):
            self.encoding[c] = one_hot[i]

    def __len__(self):
        return len(self.class_dict.keys())

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filepath, category = list(self.class_dict.items())[index]
        image = np.array(Image.open(filepath).convert('RGB'))

        image = self.tfms(image=image)['image']
        label = self.encoding[category]
        return image, label