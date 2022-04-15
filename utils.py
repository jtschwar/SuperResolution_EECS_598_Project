import matplotlib.pyplot as plt
import skimage
import torch

# Functions for Evaluation Metrics 
def psnr(img, truth):
    return skimage.metrics.peak_signal_noise_ratio(truth, img)
    # return 20 * torch.log10(torch.max(truth)) - 10 * torch.log10(torch.mean((img-truth)**2))

def ssim(img, truth):
    return skimage.metrics.structural_similarity(truth, img)

# Color Space Conversions
def rgb2ycbcr(in_img):
    return skimage.color.rgb2ycbcr(in_img)

def ycbcr2rgb(in_img):
    return skimage.color.ycbcr2rgb(in_img)

# Visualize Patches / Low Resolution Images (for Notebook)