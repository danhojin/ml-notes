from pathlib import Path
from PIL import Image, ImageChops

import numpy as np
from functools import partial, lru_cache

import torch
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from torch.utils.data import Dataset


class Generate5DigitDataset(Dataset):
    
    def __init__(self, data_home: Path, set_size=20000, transform=None):
        self.dh = data_home
        self.set_size = set_size
        self.transform = transform
        
        self.mnist = MNIST(self.dh, transform=self._mnist_transform, download=True)
        self.digit_n = np.array([1, 2, 3, 4, 5])
        self.digit_p = np.array([5, 20, 30, 30, 15])  # n digit prob.
        self.digit_p = self.digit_p / self.digit_p.sum()
        self.np_shape = (28, 150)
        self.wh = (150, 28)  # PIL shape Width x Height

    
    def _resize(self, image):
        base = Image.new('L', self.wh)
        base.paste(image)
        return base
    
    def _mnist_transform(self, image):
        transform = transforms.Compose([
            partial(lambda padding, img: transforms.functional.pad(img, padding), 4),
            transforms.RandomRotation(10),
            transforms.RandomAffine(10),
            transforms.RandomResizedCrop((28, 28), scale=(0.6, 1.0), ratio=(0.75, 1.3)),
            self._resize,
        ])
        return transform(image)
    
    def __len__(self):
        return self.set_size
    
    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
            
        n_digit = np.random.choice(self.digit_n, 1, p=self.digit_p)
        offsets = np.concatenate([
            np.random.choice(30, 1),
            np.random.choice(np.arange(16, 29), n_digit - 1)
        ])
        offsets = np.add.accumulate(offsets)
        
        image = np.zeros(self.np_shape, dtype=np.int64)
        label = []
        for i in range(n_digit.item()):
            img, lbl = self.mnist[np.random.choice(len(self.mnist), 1).item()]
            img = ImageChops.offset(img, xoffset=offsets[i], yoffset=0)
            img = np.array(img, dtype=np.int64)
            image = image + img
            label.append(lbl)
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        label_len = len(label)

        if self.transform:
            image = self.transform(image)
            label = np.array(label, dtype=np.long) + 1  # 0 reserved for blank
            label = np.concatenate([
                label,
                np.array([0] * (5 - label_len), dtype=np.long)
            ], axis=-1)
            label = torch.from_numpy(label)
            
        return image, (label, label_len)
