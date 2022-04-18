from re import L
from torch.utils.data import Dataset
import numpy as np
import h5py

# Define training dataset loading methods
class trainDataset(Dataset):
    def __init__(self, h5_file, in_group):
        super(trainDataset, self).__init__()
        self.h5_file = h5_file
        self.group = in_group

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as file:
            return np.expand_dims(file['lr'][idx] / 255, 0), np.expand_dims(file['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as file:
            return len(file['lr'])

# Define testing dataset loading methods
class testDataset(Dataset):
    def __init__(self, h5_file):
        super(testDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as file:
            return np.expand_dims(file['lr'][idx] / 255, 0), np.expand_dims(file['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as file:
            return len(file['lr'])