from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch

from datasets import trainDataset, testDataset
from tqdm import tqdm
import utils


def train_model(inModel=None, train_file=None, eval_file=None,
                lr=1e-3, batch_size=16, num_epochs=100):
   
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   # torch.manual_seed(time.time())

   model = inModel.to(device)
   
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam([
      {'params': model.parameters(), 'lr': lr * 0.1}, ], lr = lr)

   train_dataset = trainDataset(train_file,'General100')
   train_dataloader = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True)

   test_dataset = testDataset(eval_file, 'Set5')
   test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=1)                                

   # Create Trianing Process log file
   # writer = SummaryWriter('logs')

   metrics = {'test_psnr': [], 'loss': []}
   best_epoch = 0
   best_psnr = 0.0
   for epoch in range(num_epochs):
      
      model.train()
      epoch_losses = utils.AverageMeter()

      with tqdm(train_dataloader, unit='batch') as tepoch:
         for input, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            optimizer.zero_grad()
            input = input.to(device); target = target.to(device)
            target.requires_grad = True

            output = model(input)

            loss = criterion(output, target)
            epoch_losses.update(loss.item(),len(input))

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())
      
      metrics['loss'].append(epoch_losses.avg)
            
      model.eval()
      epoch_psnr = utils.AverageMeter()

      for data in test_dataloader:

         inputs, targets = data
         inputs = inputs.to(device); targets = targets.to(device)

         with torch.no_grad(): preds = model(inputs)

         epoch_psnr.update(utils.psnr(preds.cpu().numpy()/255, targets.cpu().numpy()/255), len(inputs))

      metrics['test_psnr'].append(epoch_psnr.avg)
      if epoch_psnr.avg > best_psnr:
         best_epoch = epoch 
         best_psnr = epoch_psnr.avg

         # Save Best Model
         torch.save(model.state_dict(), 'results/' + model.__class__.__name__ + '.pth')

   print('best epoch: {}, psnr: {:.2f}'.format(best_epoch,best_psnr))

   return model, metrics

      