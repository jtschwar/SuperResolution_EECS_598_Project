from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob, h5py
import skimage
import torch

# import pdb; pdb.set_trace()

# Functions for Evaluation Metrics 
def psnr(img, truth):
    return skimage.metrics.peak_signal_noise_ratio(truth, img)

def ssim(img, truth):
    return skimage.metrics.structural_similarity(truth, img, channel_axis=2)

def mse(img, truth):
    return skimage.metrics.mean_squared_error(img,truth)

# Color Space Conversions
def rgb2ycbcr(in_img):
    return skimage.color.rgb2ycbcr(in_img)

def ycbcr2rgb(in_img):
    return skimage.color.ycbcr2rgb(in_img)

# Linear or Bicubic Interpolation Functions
def downsample(in_img, scale, model, order=3):

    if order == 3: resample = Image.BICUBIC
    elif order == 2: resample = Image.BILINEAR

    lr = in_img.resize((in_img.width // scale, in_img.height // scale), resample=resample)

    if model == 'SRCNN':
        lr = lr.resize((lr.width * scale, lr.height * scale), resample=resample)

    return lr

def upsample(in_img, scale, order=3): 

    if order == 3: resample = Image.BICUBIC
    elif order == 2: resample = Image.BILINEAR

    lr = Image.fromarray(np.uint8(in_img/np.max(in_img)*255))

    out_img = lr.resize((lr.width * scale, lr.height * scale), resample=resample)
    return np.array(out_img,np.float32)

# Read image files, create sub-images and save them as a h5 file format. 
def make_data(scale=3,patch_size=33,model='SRCNN',interp_order=3):
    # Scale -- value by which resizing will take place
    # Patch Size -- size of features for feature extraction
    # Model -- (perform upsampling before for preprocessing)
    # Interpretation Order -- bilinear or bicubic interpolation 

    train_folders = ['General100', 'T91', 'BSDS200']
    test_folders  = ['Set5','Set14']

    inputs = []; targets = []
    trainFile = h5py.File('data/'+model+'_train.h5', 'w')
    valFile = h5py.File('data/'+model+'_val.h5', 'w')
    for folder in train_folders:
        imageFiles = glob.glob('SR_training_datasets/'+folder+'/*png')
        sub_input, sub_target = preprocess_train(imageFiles,scale,patch_size,model,order=interp_order)
        inputs += sub_input; targets += sub_target

    # Split Training and Validation Dataset 
    inputsTrain, inputsVal, targetsTrain, targetsVal = train_test_split(inputs, targets, test_size=0.15, random_state=42)

    trainFile.create_dataset('lr',data=inputsTrain);  valFile.create_dataset('lr',data=inputsVal)
    trainFile.create_dataset('hr',data=targetsTrain); valFile.create_dataset('hr',data=targetsVal)

    trainFile.close(); valFile.close()

    testFile = h5py.File('data/'+model+'_test.h5', 'w'); evalFile = h5py.File('data/'+model+'_eval.h5', 'w')
    for folder in test_folders:
        imageFiles = glob.glob('SR_testing_datasets/'+folder+'/*png')

        inputs, targets, rawInputs, rawTargets = preprocess_eval(imageFiles,scale,model,order=interp_order)

        test_group = testFile.create_group(folder)
        lrGroup = test_group.create_group('lr'); hrGroup = test_group.create_group('hr')

        eval_group = evalFile.create_group(folder)
        rawLRgroup = eval_group.create_group('lr'); rawHRgroup = eval_group.create_group('hr')

        for i in range(len(inputs)):
            lrGroup.create_dataset(str(i),data=inputs[i]); hrGroup.create_dataset(str(i),data=targets[i])
            rawLRgroup.create_dataset(str(i),data=rawInputs[i]); rawHRgroup.create_dataset(str(i),data=rawTargets[i])

    testFile.close(); evalFile.close()    

    

def preprocess_train(imageFiles,scale,patch_size,model,order=3):
    
    if model == 'SRCNN': stride = 14 # referenced from SRCNN paper
    else: stride = scale

    lr_patches = []; hr_patches = []
    for fName in imageFiles:

        lr, hr = preprocess(fName, scale, model, order)[:2]

        # Create Patches
        for i in range(0, lr.shape[0] - patch_size + 1, stride):
            for j in range(0, lr.shape[1] - patch_size + 1, stride):
                lr_patches.append( lr[i:i+patch_size, j:j+patch_size] )
                
                if model == 'SRCNN':
                    hr_patches.append( hr[i:i+patch_size, j:j+patch_size] )
                else:
                    hr_patches.append( hr[i*scale:(i+patch_size)*scale, j*scale:(j+patch_size)*scale])

        return lr_patches, hr_patches

def preprocess_eval(imageFiles,scale,model,order=3):

    lr_imgs = []; hr_imgs = []; raw_hr = []; raw_lr = []

    for fName in imageFiles:

        lr, hr, rawLR, rawHR = preprocess(fName, scale, model, order)

        lr_imgs.append(lr); hr_imgs.append(hr) 
        raw_lr.append(rawLR); raw_hr.append(rawHR)

    return lr_imgs, hr_imgs, raw_lr, raw_hr

def preprocess(fname, scale, model, order):

    # Read High-Resolution (hr) image
    hrIm = np.array(Image.open(fname).convert('RGB'),dtype=np.float32)
    hrIm = mod_crop(hrIm,scale)

    # Normalize 
    hrIm /= np.max(hrIm)    

    # Downsample 
    hrIm = Image.fromarray(np.uint8(hrIm*255))
    lrIm = np.array(downsample(hrIm,scale,model,order), np.float32)

    hr = rgb2ycbcr(np.array(hrIm,np.float32))[:,:,0]; lr = rgb2ycbcr(lrIm)[:,:,0]

    return lr, hr, lrIm, np.array(hrIm,dtype=np.float32)

# To rescale the original image, we need to ensure there's no remainder from scaling. 
def mod_crop(image, scale=3):
    
    h, w, _ = image.shape
    
    # Find modulo of height / width and subtract from original dims
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)

    # Crop out excess 
    return image[0:h, 0:w, :]

def srcnn_prediction(model,img):

    model.eval()

    # convert to ycbc
    img = rgb2ycbcr(img)

    # apply forward model
    pred = model(torch.tensor(img[:,:,0]).reshape(1,1,img.shape[0],img.shape[1]).to('cuda'))

    # return luminance and upsample color channels
    img[:,:,0] = pred.detach().cpu()

    # convert back to rbg
    return ycbcr2rgb(img)

def srcnn_prediction_val(model,img):

    model.eval()

    # apply forward model
    pred = model(torch.tensor(img).reshape(1,1,img.shape[0],img.shape[1]).to('cuda'))

    # return luminance and upsample color channels
    img = pred.detach().cpu()

    # convert back to rbg
    return img[0,0,:,:]


def prediction(model,lr,scale=3):

    model.eval()

    # convert to ycbc
    input = rgb2ycbcr(lr)
    output = rgb2ycbcr(upsample(lr,scale))

    # apply forward model
    pred = model(torch.tensor(input[:,:,0]).reshape(1,1,input.shape[0],input.shape[1]).to('cuda'))

    # return luminance and upsample color channels
    output[:,:,0] = pred.detach().cpu() * 255

    # convert back to rbg
    return ycbcr2rgb(output)
    

def show_model_predictions(model, lr, hr, scale, val=False):

    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,4))
    ax1.imshow(hr/255); ax1.set_title('High Resolution'); ax1.axis('off')
    ax2.imshow(lr/255); ax2.set_title('Low Resolution (x{})'.format(scale)); ax2.axis('off')
    
    if model.__class__.__name__ == 'SRCNN':
        if val: pred = srcnn_prediction_val(model,lr/255)    
        else: pred = srcnn_prediction(model,lr/255)
        ax3.imshow(pred); ax3.set_title('Prediction'); ax3.axis('off')
    else:
        if val: pred = srcnn_prediction_val(model,lr/255)  
        else: pred = prediction(model,lr/255,scale)
        ax3.imshow(pred/255); ax3.set_title('Prediction'); ax3.axis('off')
    

# Computes + stores average and current values
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



    