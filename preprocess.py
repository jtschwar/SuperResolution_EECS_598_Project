from PIL import Image
import glob, h5py
import skimage
import utils

# Prepare degraded images by introducing distortions via resizing
def train(args):
    # Path -- file path where the high-res images are stored
    # Scale -- value by which resizing will take place
    train_file = h5py.File('train.h5', 'w')

    lr_patches = []; hr_patches = []

    folders = ['General100', 'BSDS200', 'T91']

    for fol in folders:
        imageFiles = glob.glob(fol+'/*png')

        for fName in imageFiles:
            hrIm = Image.open(fol+'/'+fName).convert('RGB')
            width = (hr.width // args.scale) * args.scale
            height = (hr.height // args.scale) * args.scale

            hr = hr.resize((width, height), resample=Image.BICUBIC)

            hr = utils.rgb2ycbcr(np.array(hr,dtype=np.float)

            # Add Data Augmentation ?



def validation(args):
    test_file = h5py.File('test.h5', 'w')