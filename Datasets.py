from torch.utils.data import Dataset
from skimage import io
import torch.nn
from random import randint

import os
import torch

class UnpairedData(Dataset):

    def __init__(self, A_path, B_path, BtoA = False, transform=None):

        self.transform = transform

        self.ANames = os.listdir(A_path)
        self.A_path = A_path

        self.BNames = os.listdir(B_path)
        self.B_path = B_path
        self.len_A = len(self.ANames)
        self.len_B = len(self.BNames)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= self.len_A:
            A_name = self.ANames[randint(0,self.len_A-1)]
        else:
            A_name = self.ANames[idx]

        if idx >= self.len_B:
            B_name = self.BNames[randint(0,self.len_B-1)]
        else:
            B_name = self.BNames[idx]

        img_A = io.imread(os.path.join(self.A_path,
                                A_name))
        img_A = img_A.transpose(2,0,1)
        img_A = img_A/256

        img_B = io.imread(os.path.join(self.B_path,
                                       B_name))
        img_B = img_B.transpose(2, 0, 1)
        img_B = img_B / 256

        sample = {'image_A': img_A, 'image_B': img_B}

        if self.transform:
            sample = self.transform(sample)

        return sample
