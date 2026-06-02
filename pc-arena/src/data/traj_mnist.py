import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.utils import instantiate_from_config

class TrajMNIST(Dataset):
    def __init__(self, root, train=True, transform_fns=None):
        """
        Args:
            root (str): Directory containing the preprocessed tensor files.
            train (bool): If True, loads the training set, otherwise validation.
            transform_fns (list): Optional list of transformations from configs.
        """
        self.train = train
        self.root = root
        
        # Load pre-processed data (Modify file names as needed for your setup)
        if self.train:
            data_path = os.path.join(root, "train.pt")
        else:
            data_path = os.path.join(root, "val.pt")
            
        # Assuming the data is saved as a single tensor of shape [num_samples, seq_length/features]
        self.data = torch.load(data_path)
        self.length = self.data.size(0)

        # Setup transforms if defined in the YAML config
        if transform_fns is not None:
            transforms_list = [instantiate_from_config(t) for t in transform_fns]
            self.transforms = v2.Compose(transforms_list)
        else:
            self.transforms = lambda x: x

    def __getitem__(self, index):
        # Fetch the sample
        sample = self.data[index]
        
        # Apply transforms (e.g., flattening, quantizing)
        return self.transforms(sample)

    def __len__(self):
        return self.length