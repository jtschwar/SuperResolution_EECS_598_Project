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
            return np.expand_dims(file[self.group+'/lr'][idx] / 255, 0), \
                                  np.expand_dims(file[self.group+'/hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as file:
            return len(file[self.group+'/lr'])

# Define testing dataset loading methods
class testDataset(Dataset):
    def __init__(self, h5_file, in_group):
        super(testDataset, self).__init__()
        self.h5_file = h5_file
        self.group = in_group

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as file:
            return np.expand_dims(file[self.group+'/lr_'+str(idx)] / 255, 0), \
                                  np.expand_dims(file[self.group+'/hr_'+str(idx)] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as file:
            return len(file['lr'])


# class H5_Dataset(Dataset):
#     def __init__(self, hd5_file, database_name):
#         self.database_name = database_name
#         self.h5_file = h5py.File(hd5_file, 'r')
#         super().__init__()
#         self.index_dict = self.__register_indexes()
 
#     def __register_indexes(self):
#         """
#         Returns a dictionary that assigns index
#         to image location on the hdf5 file
#         """
#         index_dict = {}
#         counter = 0
#         for group, group_db in self.h5_file.items():
#             for i in range(len(group_db[self.database_name])):
#                 index_dict[counter] = {'group': group, 'index': i}
#                 counter += 1
#         return index_dict
 
#     def __getitem__(self, index):
#         item_info = self.index_dict[index]
#         group = item_info['group']
#         group_index = item_info['index']
#         return {'group': group, 'image': self.h5_file[group][self.database_name][group_index]}
#         # I havent written something about the labels,
#         # because I don't know how your file is structured
 
#     def __len__(self):
#         return len(self.index_dict)
