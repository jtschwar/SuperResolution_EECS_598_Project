import matplotlib.pyplot as plt
from enum import Enum
from PIL import Image
import scipy.ndimage
import numpy as np
import glob, h5py
import skimage

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

# Read image files, create sub-images and save them as a h5 file format. 
def make_data(scale=3,patch_size=33,model='SRCNN'):
    # Scale -- value by which resizing will take place

    train_folders = ['General100', 'T91']
    test_folders  = ['Set5','Set14']

    print('Creating Training Dataset..')
    file = h5py.File(model+'_train.h5', 'w')
    for folder in train_folders:
        imageFiles = glob.glob('SR_training/'+folder+'/*png')

        sub_input, sub_label = preprocess_train(imageFiles,scale,patch_size,model)

        group = file.create_group(folder)
        group.create_dataset('lr',data=sub_input)
        group.create_dataset('hr',data=sub_label)

    print('Creating Testing Dataset..')
    file = h5py.File(model+'_test.h5', 'w')
    for folder in test_folders:
        imageFiles = glob.glob('SR_testing/'+folder+'/*png')

        sub_input, sub_label = preprocess_eval(imageFiles,scale,model)

        group = file.create_group(folder)
        for i in range(len(sub_input)):
            group.create_dataset('lr_'+str(i),data=sub_input[i])
            group.create_dataset('hr_'+str(i),data=sub_label[i])
    

def preprocess_train(imageFiles,scale,patch_size,model):
    
    lr_patches = []; hr_patches = []
    
    for fName in imageFiles:

        # Read High-Resolution (hr) image
        hrIm = np.array(Image.open(fName).convert('RGB'),dtype=np.float32)
        hrIm = mod_crop(hrIm,scale)

        # Normalize 
        hrIm /= 255

        lrIm = scipy.ndimage.interpolation.zoom(hrIm, (1/scale,1/scale,1),prefilter=False)
        
        # Upsample LR image (necessary preprocessing step for SRCNN)
        if model == 'SRCNN':
            lrIm = scipy.ndimage.interpolation.zoom(lrIm, (scale,scale,1),prefilter=False)

        # Extract y-channel (luminance)
        hr = rgb2ycbcr(np.array(hrIm,dtype=np.float32))[:,:,0]
        lr = rgb2ycbcr(np.array(lrIm,dtype=np.float32))[:,:,0]

        # Create Patches
        for i in range(0, lr.shape[0] - patch_size + 1, scale):
            for j in range(0, lr.shape[1] - patch_size + 1, scale):
                lr_patches.append( lr[i:i+patch_size, j:j+patch_size] )
                
                if model == 'SRCNN':
                    hr_patches.append( hr[i:i+patch_size, j:j+patch_size] )
                else:
                    hr_patches.append( hr[i*scale:(i+patch_size)*scale, j*scale:(j+patch_size)*scale])

        # Add Data Augmentation ??

        return hr_patches, lr_patches

def preprocess_eval(imageFiles,scale,model):

    lr_imgs = []; hr_imgs = []

    for fName in imageFiles:

        # Read High-Resolution (hr) image
        hrIm = np.array(Image.open(fName).convert('RGB'),dtype=np.float32)
        hrIm = mod_crop(hrIm,scale)

        # Normalize 
        hrIm /= 255

        lrIm = scipy.ndimage.interpolation.zoom(hrIm, (1/scale,1/scale,1),prefilter=False)
        
        # Upsample LR image (necessary preprocessing step for SRCNN)
        if model == 'SRCNN':
            lrIm = scipy.ndimage.interpolation.zoom(lrIm, (scale,scale,1),prefilter=False)

        hr = rgb2ycbcr(np.array(hrIm,dtype=np.float32))[:,:,0]
        lr = rgb2ycbcr(np.array(lrIm,dtype=np.float32))[:,:,0]

        lr_imgs.append(lr); hr_imgs.append(hr)

    return hr_imgs, lr_imgs


# To rescale the original image, we need to ensure there's no remainder from scaling. 
def mod_crop(image, scale=3):
    
    h, w, _ = image.shape
    
    # Find modulo of height / width and subtract from original dims
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)

    # Crop out excess 
    return image[0:h, 0:w, :]

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

# Computes + stores average and current values
class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


    