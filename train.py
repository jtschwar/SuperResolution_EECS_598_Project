from torch.utils.data.dataloader import DataLoader
from torch import nn
import torch

from datasets import trainDataset, testDataset
from tqdm import tqdm
import utils


def train_model(inModel=None, lr=1e-3, batch_size=16, num_epochs=100):
   
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   # torch.manual_seed(time.time())

   model = inModel.to(device)
   modelName = model.__class__.__name__
   
   criterion = nn.MSELoss()
   # criterion = nn.L1Loss() # ran a few tests on L1 loss, the improvement wasn't dramatic..

   optimizer = torch.optim.Adam([
      {'params': model.parameters(), 'lr': lr * 0.1}, ], lr = lr)

   # Dataloaders for Training, Validation and Test (Set5 and 14)
   train_dataloader = DataLoader(dataset= trainDataset(modelName+'_train.h5'),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True)

   set5_test_dataloader = DataLoader(dataset=testDataset(modelName+'_test.h5', 'Set5'),
                                batch_size=1)

   set14_test_dataloader = DataLoader(dataset=testDataset(modelName+'_test.h5', 'Set14'),
                                batch_size=1)                                

   val_dataloader = DataLoader(dataset=trainDataset(modelName+'_val.h5'), batch_size=1)

   metrics = {'set5_psnr': [], 'set14_psnr': [], 'val_psnr': [],
              'set5_ssim': [], 'set14_ssim': [], 'val_ssim': [], 'loss': [], }

   best_epoch = 0; best_psnr = 0.0
   for epoch in tqdm(range(num_epochs), desc='Epoch'):
      
      model.train()
      epoch_losses = utils.AverageMeter()

      for _, data in enumerate(train_dataloader,0):

            input, target = data

            optimizer.zero_grad()
            input = input.to(device); target = target.to(device)
            target.requires_grad = True

            output = model(input)

            loss = criterion(output, target)
            epoch_losses.update(loss.item(),len(input))

            loss.backward()
            optimizer.step()
      
      metrics['loss'].append(epoch_losses.avg)
            
      model.eval()

      metrics = eval(model, val_dataloader, device, metrics, 'val')[0]
      metrics = eval(model, set14_test_dataloader, device, metrics, 'set14')[0]
      metrics, epoch_psnr = eval(model, set5_test_dataloader, device, metrics, 'set5')

      # Save Best Model
      if epoch_psnr.avg > best_psnr:
         best_epoch = epoch 
         best_psnr = epoch_psnr.avg
         torch.save(model.state_dict(), 'results/' + model.__class__.__name__ + '.pth')

   print('best epoch: {}, psnr: {:.2f}'.format(best_epoch,best_psnr))

   return model, metrics

      
def eval(model, dataloader, device, metrics, name):

   epoch_psnr = utils.AverageMeter(); epoch_ssim = utils.AverageMeter()

   for data in dataloader:
      inputs, targets = data
      inputs = inputs.to(device); targets = targets.to(device)

      with torch.no_grad(): preds = model(inputs)

      epoch_psnr.update(utils.psnr(preds.cpu().numpy()/255, targets.cpu().numpy()/255), len(inputs))     
      epoch_ssim.update( utils.ssim(torch.swapaxes(preds[0,],0,2).cpu().numpy()/255, torch.swapaxes(targets[0,],0,2).cpu().numpy()/255), len(inputs)) 

      metrics[name+'_psnr'].append(epoch_psnr.avg); metrics[name+'_ssim'].append(epoch_ssim.avg)

      return metrics, epoch_psnr


