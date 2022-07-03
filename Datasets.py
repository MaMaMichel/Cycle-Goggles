from torch.utils.data import Dataset
from skimage import io
import torch.nn

import os
import torch

class DancingDataset(Dataset):

    # TODO  figure out dataloading

    def __init__(self, orig_path, transform=None):

        self.transform = transform

        self.imageNames = os.listdir(orig_path)
        self.orig_path = orig_path

    def __len__(self):
        return len(self.imageNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.imageNames[idx]

        img_orig = io.imread(os.path.join(self.orig_path,
                                img_name))
        img_orig = img_orig.transpose(2,0,1)
        img_orig = img_orig/256

        sample = {'image': img_orig}

        if self.transform:
            sample = self.transform(sample)

        return sample
