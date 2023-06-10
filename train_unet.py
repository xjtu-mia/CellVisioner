import os
import numpy as np
import random
import time
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from networks.muti_task_model import UNet

from source.dataset import mask_muti_channel_Dataset
from predict import predict
import matplotlib.pyplot as plt
from collections import OrderedDict
from source.utils import PearsonCorrelation, LambdaLR
from source.eval import eval_net
from options import TrainOptions
import pandas as pd

os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

num_workers = 4
buffered_in_memory = True

def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


def train(opts, train_path,val_path,  result_path, net):
    # SingleDataset
    train_dataset = mask_muti_channel_Dataset(train_path, 'train', opts, opts.augment)
    # val_dataset = mask_muti_channel_Dataset(val_path, 'val', opts, False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                   num_workers=num_workers)
    # val_loader = data.DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=True,
    #                              num_workers=num_workers)
 
    criterion = torch.nn.MSELoss()
    
    # writer = SummaryWriter(os.path.join(result_path, 'log'))
    optimizer = optim.Adam(net.parameters(),
                           lr=opts.learn_rate_u,
                           weight_decay=0)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opts.epoch_num, 0,
                                                                             opts.epoch_num-10).step)  # sum * epoch,offset,start delay epoch
    
    
    best_val_loss = 1000.0
    best_loss_epoch = 0
    val_loss = 0
    if opts.load_weights:
        net.load_state_dict(torch.load(
            os.path.join(opts.weights_path, 'model_best_loss.pth'))['model_state_dict'])
        print('loading weights')
    
    # else:
    #     initial_net(net)
    
    train_ids = len(train_loader)
    train_loss_history = []
    val_loss_history = []
    ssim_loss_history = []
    psnr_loss_history = []
    
    for epoch in range(opts.epoch_num):
        print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        t0 = time.perf_counter()
        # print('Starting epoch {}/{}.'.format(epoch + 1, epoch_num))
        net.train()
        train_loss = 0
        train_sample_num = 0
        # training for one epoch
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{opts.epoch_num}', unit='img') as pbar:
            for batch_id, (imgs, gt_mask, mask, _) in enumerate(train_loader):
                # fetch data
                n = len(imgs)
                # convert to GPU memory
                imgs = imgs.cuda()
                gt_mask = gt_mask.cuda()
                mask = mask.cuda()
                prob = net(imgs)
                # compute loss
                loss= criterion(prob, gt_mask)
                train_loss += n * loss.item()
                train_sample_num += n
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                # backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_id > 112:
                    break
        lr_scheduler.step()
        train_loss = train_loss / train_sample_num
        print('Epoch {0:d}/{1:d}---Finished'.format(epoch + 1, opts.epoch_num))
        print('Training loss: {:.6f}'.format(train_loss))
        
        # validation
        val_save_dir = result_path + '/val'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        # val_loss, ssim, psnr = eval_net(opts, outdir=val_save_dir, net=net, loader=val_loader
        #                                 , epoch=epoch, criterion=criterion)
        # print('Val loss,ssim,psnr: {:.6f}- {:.6f}- {:.6f}'.format(val_loss, ssim, psnr))
        
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('ssim', ssim, epoch)
        # writer.add_scalar('psnr', psnr, epoch)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        # ssim_loss_history.append(ssim)
        # psnr_loss_history.append(psnr)
        

        # if epoch+1==opts.epoch_num:
        
        # # if epoch == 0 or 1.0 > val_loss > best_val_loss:
        # if epoch == 0 or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     with open(result_path + r'/val_loss.txt', 'w') as f:
        #         f.writelines(f'{epoch}:{best_val_loss}' + '\n')
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "epoch": epoch,
                "epoch_loss": best_val_loss,
            },
            os.path.join(result_path, f"model_best_loss.pth"),
        )
           
        
        # else:
        #     print(f'val loss not improve from epoch:{best_loss_epoch},best_val_loss:{best_val_loss}!')
        
        # plot loss
        # epochs = range(1, len(train_loss_history) + 1)
        # plt.plot(epochs, train_loss_history, 'y', label='Training loss')
        # plt.plot(epochs, val_loss_history, 'r', label='Validation loss')
        # plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('loss')
        
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys())
        # # plt.legend()
        # plt.savefig(result_path + '/loss.png')
    
    # plt.close()
    # epochs = range(1, len(train_loss_history) + 1)
    # df = pd.DataFrame({'epochs': epochs, 'loss': train_loss_history,
    #                    })
    # save_path = os.path.join(result_path, f'loss.csv')
    # df.to_csv(path_or_buf=save_path, sep=',', na_rep='NA',
    #           columns=['epochs', 'loss'])
    # predict(opts, val_path, result_path, net)


def write_opts(opts, sava_path):
    argsDict = opts.__dict__
    with open(sava_path + r'/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def unet_main(user_name,fluorescence,weight_name):
    parser = TrainOptions()
    opts = parser.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dataroot = opts.dataroot
    save_dir = opts.save_dir
    pb_aug = opts.pb_aug
    model_name = 'unet'
    cell_and_magnification = user_name
    weights_path=fr'./model_weight/unet_{fluorescence}/{weight_name}'
    opts.weights_path=weights_path
    print(opts.weights_path)
    opts.flu=fluorescence
    print(opts.flu)
    train_path = os.path.join(dataroot, cell_and_magnification, 'pbda')
    val_path = os.path.join(dataroot, cell_and_magnification, 'val')
    result_path = os.path.join(save_dir, cell_and_magnification,
                               fluorescence, model_name, pb_aug, f'result0')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    write_opts(opts, result_path)
    net = UNet(in_ch=opts.inch)
    net.cuda()
    train(opts, train_path,val_path ,result_path, net=net)

if __name__ == "__main__":
    parser = TrainOptions()
    opts = parser.parse()
    
    
