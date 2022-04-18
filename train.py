from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch


from datasets import trainDataset, testDataset
from tqdm import tqdm
import models, utils


def train_model(inModel=None, train_file=None, eval_file=None,
                scale=2, lr=1e-3, batch_size=16, num_epochs=100):
   
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   model = inModel.to(device)
   
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam([])

   train_dataloader = DataLoader(dataset=trainDataset(train_file),
                                 batch_size=batch_size,
                                 shuffle=True)
   test_dataloader = DataLoader(dataset=testDataset(eval_file),
                                batch_size=1)

   # Create Trianing Process log file
   writer = SummaryWriter()

   best_psnr = 0.0
   for epoch in range(num_epochs):
      
      model.train()
      epoch_losses = utils.AverageMeter()

      # with tqdm(total=(len(train)))

      for data in train_dataloader:

         inputs, labels = data
         inputs = inputs.to(device); labels = labels.to(device)
         
         preds = model(inputs)

         loss = criterion(preds, labels)

         epoch_losses.update(loss.time(),len(inputs))

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

      model.eval()
      epoch_psnr = utils.AverageMeter()

      for data in test_dataloader:

         inputs, labels = data
         inputs = inputs.to(device); labels = labels.to(device)

         with torch.no_grad(): pred = model(inputs).clamp(0.0,1.0)

         epoch_psnr.update(utils.psnr(preds, labels), len(inputs))

      # epoch_losses ?

      