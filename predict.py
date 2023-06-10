import sys
import os
from optparse import OptionParser
import numpy as np
import random
import time
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from options import TrainOptions
from torch.utils import data
from source.dataset import testDataset
from source.utils import sample_image_test
from networks.cGAN_drop import Unet as G_Unet_drop
from networks.muti_task_model import UNet as UNet

num_workers = 4
buffered_in_memory = True



def infer_full_image(net, input, C_out, kernel_size=256, stride=128):
    B, C, W, H = input.shape
    pad_W = kernel_size - W % kernel_size
    pad_H = kernel_size - H % kernel_size
    x = compute_pyramid_patch_weight_loss(kernel_size, kernel_size)
    input = F.pad(input, (0, pad_H, 0, pad_W), mode="reflect").squeeze(0)  # 进行镜像填充，左下 0，右上 pad_H, pad_W
    _, W_pad, H_pad = input.shape
    patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)  # 滑动整张图，窗口大小256，步长128
    c, n_w, n_h, w, h = patches.shape
    patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)
    dataset = torch.utils.data.TensorDataset(patches.permute(1, 0, 2, 3))
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    op = []
    for batch_idx, sample1 in enumerate(dataloader):
        patch_op = net(sample1[0])
        op.append(patch_op)
    op = torch.cat(op).permute(1, 0, 2, 3)
    op = op.permute(0, 2, 3, 1).reshape(1, -1, n_w * n_h)
    weights_op = (
        torch.from_numpy(x)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(1, C_out, 1, n_w * n_h)
            .reshape(1, -1, n_w * n_h) ).cuda()
   
    op = torch.mul(weights_op, op)
    op = F.fold(
        op,
        output_size=(W_pad, H_pad),
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
    )
    weights_op = F.fold(
        weights_op,
        output_size=(W_pad, H_pad),
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
    )
    output = torch.divide(op, weights_op)
    output = output[:, :, :W, :H]
    return output


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    xc = width * 0.5
    yc = height * 0.5
    xl = 0;xr = width;yb = 0;yt = height
    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)
    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)
    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)
    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W

def predict(opts,test_path,result_path, net,model_path=None):
    if model_path==None:
        model_path=result_path
    best_model_filename = model_path + r'/model_best_loss.pth'
    net.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
    test_dataset =testDataset(test_path,opts,augment=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                 num_workers=num_workers)
    net.eval()
    for it, (img,img_name) in enumerate(test_loader):
        img = img.cuda()
        with torch.no_grad():
                C_out = img.shape[1]
                prob = infer_full_image(net,img, C_out, kernel_size=256, stride=128)
        sample_image_test(opts,result_path, ''.join(img_name), prob)

def predict_main(user_name,fluorescence,model_name):
    parser = TrainOptions()
    opts = parser.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net=UNet(in_ch=opts.inch) if model_name=='unet' else G_Unet_drop(in_ch=opts.inch)
    net.cuda()
    dataroot = './datasets'
    save_dir = './results'
    pb_aug = opts.pb_aug
    test_path = os.path.join(dataroot, user_name, 'test_bright')
    result_path = os.path.join(save_dir, user_name,
                               fluorescence, model_name, pb_aug, f'result0')
    predict(opts, test_path, result_path, net, result_path)
if __name__=="__main__":
    parser = TrainOptions()
    opts = parser.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # net=cgan_unet_drop(in_ch=5)
    net=UNet(in_ch=5)
    # net = Unet(in_ch=5)
    # net = nn.DataParallel(module=model)
    net.cuda()
    dataroot = r'E:\FRM_floder/datasets'
    save_dir = './result'
    pb_aug = opts.pb_aug
    model_name = opts.model_name

    cell_and_magnification = opts.cell_and_magnification
    val_path = os.path.join(dataroot, cell_and_magnification, 'val_and_test')
    result_path=r'E:\FRM_floder\result\231_20X\5stack\few_images\16\actin\unet\pbda\result0'
    model_path=r'E:\FRM_floder\result\231_20X\5stack\few_images\16\actin\unet\pbda\result0'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    predict(opts,val_path,result_path, net,model_path)
    # mt_predict(opts,val_path,result_path, net)
    