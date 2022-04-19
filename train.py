from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch

from datasets import trainDataset, testDataset
from tqdm import tqdm
import utils
import time


def train_model(inModel=None, train_file=None, eval_file=None,
                scale=2, lr=1e-3, batch_size=16, num_epochs=100):
   
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   # torch.manual_seed(time.time())

   model = inModel.to(device)
   
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam([
      {'params': model.parameters(), 'lr': lr * 0.1}, ], lr = lr)

   train_dataloader = DataLoader(dataset=trainDataset(train_file,'General100'),
                                 batch_size=batch_size,
                                 shuffle=True)
   test_dataloader = DataLoader(dataset=testDataset(eval_file,'General100'),
                                batch_size=1)

   # Create Trianing Process log file
   # writer = SummaryWriter('logs')

   best_epoch = 0
   best_psnr = 0.0
   for epoch in range(num_epochs):
      
      model.train()
      epoch_losses = utils.AverageMeter()

      with tqdm(train_dataloader, unit='batch') as tepoch:
         for input, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            input = input.to(device); target = target.to(device)
            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, target)
            epoch_losses.update(loss.time(),len(input))

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
            
      model.eval()
      epoch_psnr = utils.AverageMeter()

      for data in test_dataloader:

         inputs, targets = data
         inputs = inputs.to(device); targets = targets.to(device)

         with torch.no_grad(): preds = model(inputs).clamp(0.0,1.0)

         epoch_psnr.update(utils.psnr(preds, targets), len(inputs))

      if epoch_psnr.avg > best_psnr:
         best_epoch = epoch 
         best_psnr = epoch_psnr.avg

         torch.save(model.state_dict(), 'model.pth')

      # epoch_losses ?
   print('best epoch: {}, psnr: {:.2f}'.format(best_epoch,best_psnr))

      