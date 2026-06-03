import numpy as np
import torch
import os
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import v2
from torchvision import datasets

from src.utils import instantiate_from_config


class CIFAR(Dataset):
    def __init__(self, root = "/scratch/anji/data/CIFAR/", train = True, download = True, transform_fns = None):
        # Load the original CIFAR dataset
        self.cifar = datasets.CIFAR10(
            root = root, 
            train = train, 
            download = download
        )

        if transform_fns is not None:
            transforms = []
            for transform_fn in transform_fns:
                transforms.append(instantiate_from_config(transform_fn))
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = lambda x: x

    def __getitem__(self, index):
        # Get the image and label from the original dataset
        img, label = self.cifar[index]

        img = np.array(img)
        img = torch.from_numpy(img).type(torch.uint8).float() / 127.5 - 1 # Normalize to [-1, 1]
        img = img.permute(2, 0, 1)
        sample = {"img": img, "label": label}

        return self.transforms(sample)

    def __len__(self):
        return len(self.cifar)