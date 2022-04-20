from PIL import Image
import scipy.ndimage
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

def downsample(in_img, scale, model, order=3):
    
    # Need to replace with PIL... (scipy bad)
    # lrIm = scipy.ndimage.interpolation.zoom(in_img, (1/scale,1/scale,1), order=order, prefilter=False)        

    if order == 3: resample = Image.BICUBIC
    elif order == 2: resample = Image.BILINEAR
    
    lrIm = in_img.resize((in_img.width//scale,in_img.height//scale),resample=resample)

    # Upsample LR image (necessary preprocessing step for SRCNN)
    if model == 'SRCNN':
        lrIm = in_img.resize((in_img.width*scale,in_img.height*scale),resample=resample)
        # lrIm = scipy.ndimage.interpolation.zoom(lrIm, (scale,scale,1), order=order, prefilter=False)

    return lrIm

def downsample_PIL(in_img, scale, model, order=3):

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
def make_data(scale=3,patch_size=33,model='SRCNN'):
    # Scale -- value by which resizing will take place

    train_folders = ['General100', 'T91', 'BSDS200']
    test_folders  = ['Set5','Set14']

    inputs = []; targets = []
    print('Creating Training Dataset..')
    file = h5py.File('data/'+model+'_train.h5', 'w')
    for folder in train_folders:
        imageFiles = glob.glob('SR_training_datasets/'+folder+'/*png')
        sub_input, sub_target = preprocess_train(imageFiles,scale,patch_size,model)
        inputs += sub_input; targets += sub_target

    # group = file.create_group(folder)
    file.create_dataset('lr',data=inputs)
    file.create_dataset('hr',data=targets)

    print('Creating Testing Dataset..')
    test_file = h5py.File('data/'+model+'_test.h5', 'w')
    eval_file = h5py.File('data/'+model+'_eval.h5', 'w')
    for folder in test_folders:
        imageFiles = glob.glob('SR_testing_datasets/'+folder+'/*png')

        inputs, targets, rawInputs, rawTargets = preprocess_eval(imageFiles,scale,model)

        test_group = test_file.create_group(folder)
        lrGroup = test_group.create_group('lr')
        hrGroup = test_group.create_group('hr')

        eval_group = eval_file.create_group(folder)
        rawLRgroup = eval_group.create_group('lr')
        rawHRgroup = eval_group.create_group('hr')

        for i in range(len(inputs)):
            lrGroup.create_dataset(str(i),data=inputs[i])
            hrGroup.create_dataset(str(i),data=targets[i])
            rawLRgroup.create_dataset(str(i),data=rawInputs[i])
            rawHRgroup.create_dataset(str(i),data=rawTargets[i])

    

def preprocess_train(imageFiles,scale,patch_size,model):
    
    if model == 'SRCNN': stride = 14 # referenced from SRCNN paper
    else: stride = scale

    lr_patches = []; hr_patches = []
    for fName in imageFiles:

        # Read High-Resolution (hr) image
        hrIm = np.array(Image.open(fName).convert('RGB'),dtype=np.float32)
        hrIm = mod_crop(hrIm,scale)

        # Normalize 
        hrIm /= np.max(hrIm)

        # Downsample Images by Scale
        hrIm = Image.fromarray(np.uint8(hrIm*255))
        lrIm = downsample_PIL(hrIm,scale,model)

        # Extract y-channel (luminance)
        hr = rgb2ycbcr(np.array(hrIm,dtype=np.float32))[:,:,0]
        lr = rgb2ycbcr(np.array(lrIm,dtype=np.float32))[:,:,0]

        # Create Patches
        for i in range(0, lr.shape[0] - patch_size + 1, stride):
            for j in range(0, lr.shape[1] - patch_size + 1, stride):
                lr_patches.append( lr[i:i+patch_size, j:j+patch_size] )
                
                if model == 'SRCNN':
                    hr_patches.append( hr[i:i+patch_size, j:j+patch_size] )
                else:
                    hr_patches.append( hr[i*scale:(i+patch_size)*scale, j*scale:(j+patch_size)*scale])

        return lr_patches, hr_patches

def preprocess_eval(imageFiles,scale,model):

    lr_imgs = []; hr_imgs = []; raw_hr = []; raw_lr = []

    for fName in imageFiles:

        # Read High-Resolution (hr) image
        hrIm = np.array(Image.open(fName).convert('RGB'),dtype=np.float32)
        hrIm = mod_crop(hrIm,scale)

        # # Normalize 
        hrIm /= np.max(hrIm)

        # Downsample 
        hrIm = Image.fromarray(np.uint8(hrIm*255))
        lrIm = np.array(downsample_PIL(hrIm,scale,model), np.float32)

        hr = rgb2ycbcr(np.array(hrIm,np.float32))[:,:,0]; lr = rgb2ycbcr(lrIm)[:,:,0]

        lr_imgs.append(lr); hr_imgs.append(hr) 
        raw_lr.append(lrIm); raw_hr.append(hrIm)

        # raw_imgs.append(np.array(np.stack((hrIm,lrIm)),np.float32))

    return lr_imgs, hr_imgs, raw_lr, raw_hr


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
    pred = model(torch.tensor(img[:,:,0]).reshape(1,1,img.shape[0],img.shape[1]).to('cuda'))#.clamp(1.0,0)

    # return luminance and upsample color channels
    img[:,:,0] = pred.detach().cpu()

    # convert back to rbg
    return ycbcr2rgb(img)

def prediction(model,lr,scale=3):

    model.eval()

    # convert to ycbc
    input = rgb2ycbcr(lr)
    output = rgb2ycbcr(upsample(lr,scale))

    # apply forward model
    pred = model(torch.tensor(input[:,:,0]).reshape(1,1,input.shape[0],input.shape[1]).to('cuda'))#.clamp(0,1.0)

    # return luminance and upsample color channels
    output[:,:,0] = pred.detach().cpu() * 255

    # convert back to rbg
    return ycbcr2rgb(output)
    # return pred.detach().cpu()[0,0,]
    

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



    