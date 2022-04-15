from torch.utils.tensorboard import SummaryWriter
import models
# def create_h5py_file

from torch.utils import data

num_epochs = 50
loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 6}

dataset = HDF5Dataset('C:/ml/data', recursive=True, load_data=False, 
   data_cache_size=4, transform=None)

data_loader = data.DataLoader(dataset, **loader_params)

for i in range(num_epochs):
   for x,y in data_loader:
      # here comes your training loop
      pass