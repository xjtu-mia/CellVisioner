from torchvision.utils import save_image,make_grid
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os
from collections import deque
import pandas as pd
import math
import csv


from skimage.metrics import structural_similarity as  compare_ssim
def PSNR(im1, im2):
    im1 = im1.astype(np.float64) / 255
    im2 = im2.astype(np.float) / 255
    mse = np.mean((im1 - im2)**2)
    return 10*math.log10(1. / mse)

def lamb_gan(epoch,adv_loss_weight):
    if epoch>20:
        lamb=0
    else:
        lamb=adv_loss_weight
    return lamb

def denorm(img,precess,img_type):
    #convert [-1,1] to [0,1]
    #to use torchvision.utils.save_image
    # if self.cell_type == '231_20X':
    #     preprocess_stats = [8.68216283, 27.70221643]  # 231_20X
    # elif self.cell_type == '3T3_20X':
    #     preprocess_stats = [6.81329184, 27.16556653]  # 3T3_20X
    # elif self.cell_type == '3T3_10X':
    #     preprocess_stats = [3.30066417, 18.28308778]  # 3T3_10X
    # elif self.cell_type == 'HUVEC_40X':
    #     preprocess_stats = [17.64951158, 34.55002174]
    # elif self.cell_type == '3T3_40X':
    #     preprocess_stats = [5.34233326, 20.61875249]
    preprocess_stats = [17.64951158, 34.55002174]
    mean=preprocess_stats[0]
    std=preprocess_stats[1]
    img = make_grid(img, nrow=8, padding=0, pad_value=0,).detach()
    if precess=="standard":
        img=((img * std) + mean).cpu().clip(min=0).numpy().transpose(1, 2, 0)
        nucmap_zero = np.zeros(img.shape, img.dtype)
        cv2.normalize(img, nucmap_zero, 0, 255, cv2.NORM_MINMAX)
        img = nucmap_zero.astype('uint8')
    else:
        if img_type=='label':
            img = ((img+0.0)*255+0.5).cpu().clip(min=0,max=255).numpy().transpose(1, 2, 0).astype('uint8')
        else:
            img = ((img + 1.0)*127.5+0.5).cpu().clip(min=0,max=255).numpy().transpose(1, 2, 0).astype('uint8')
    return img

    
def sample_image(opts,exp_name,epoch, actin,actin_pred):
    sample_image_dir = exp_name
    if opts.is_norm:
        precess = 'norm'
    else:
        precess = 'standard'
    img_type='label'
    if(not os.path.exists(sample_image_dir)):
        os.makedirs(sample_image_dir)
    cv2.imwrite( '{}/gt-{}.jpg'.format(sample_image_dir, epoch + 1),denorm(actin,precess,img_type))
    cv2.imwrite( '{}/pred-{}.jpg'.format(sample_image_dir, epoch + 1),denorm(actin_pred,precess,img_type))
    



def sample_image_test(opts,exp_name, imgname,pred):
    pred_sample_image_dir = os.path.join(exp_name,'test', 'pred')
    gt_sample_image_dir = os.path.join(exp_name, 'test','gt')
    if opts.is_norm:
        precess = 'norm'
    else:
        precess = 'standard'
    img_type = 'label'
    if (not os.path.exists(pred_sample_image_dir)):
        os.makedirs(pred_sample_image_dir)
    cv2.imwrite(os.path.join(pred_sample_image_dir, imgname), denorm(pred, precess, img_type))


    
    
def SSIM(img1, img2):
    (score, diff) = compare_ssim(img1, img2, multichannel=False, gaussian_weights=True, data_range=255,
                                 full=True)  #  windows size=101 , gaussian_weights=True , win_size=11  data_range=255,
    return score



class PearsonCorrelation(nn.Module):
    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost
    
    
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)



class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch,warm_up=False):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        self.warm_up=warm_up

    def step(self, epoch):
        # if epoch<10 and self.warm_up:
        #     return (epoch/10)**2
        # else:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)



