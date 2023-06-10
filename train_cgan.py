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
from networks.cGAN import Discriminator
from networks.cGAN_drop import Unet as Unet_drop

from source.dataset import mask_muti_channel_Dataset
from source.eval import eval_net
from predict import predict
import matplotlib.pyplot as plt
from collections import OrderedDict
from source.utils import  LambdaLR
import pandas as pd
from options import TrainOptions


os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=True
num_workers = 4
buffered_in_memory = True



def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)



def train(opts, train_path, result_path, model_G, model_D):
    # SingleDataset
    train_dataset = mask_muti_channel_Dataset(train_path, 'train', opts, opts.augment)
    # val_dataset = muti_channel_Dataset(val_path, 'val', opts, False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                   num_workers=num_workers)
    # val_loader = data.DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=True,
    #                              num_workers=num_workers)
    criterion = torch.nn.functional.smooth_l1_loss
    D_loss=torch.nn.MSELoss()
    writer = SummaryWriter(os.path.join(result_path, 'log'))
    optimizer_G = optim.Adam(model_G.parameters(),
                           lr=opts.learn_rate_g,
                           betas=(0.9, 0.999),
                           weight_decay=0)
    optimizer_D = optim.Adam(model_D.parameters(),
                           lr=opts.learn_rate_d,
                           betas=(0.9, 0.999),
                           weight_decay=0)
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opts.epoch_num, 0,
                                                                             opts.epoch_num-10,opts.warm_up).step)
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opts.epoch_num, 0,
                                                                             opts.epoch_num-10,opts.warm_up).step)
    val_loss = 0
    if opts.load_weights:
        model_G.load_state_dict(torch.load(
            os.path.join(opts.weights_path, 'model_best_loss.pth'))['model_state_dict'])
        model_D.load_state_dict(torch.load(
            os.path.join(opts.weights_path, 'discriminator_model_best_loss.pth'))['model_state_dict'])
    else:
        initial_net(model_G)
        initial_net(model_D)
    
    train_ids = len(train_loader)
    train_g_loss_history = []
    train_d_loss_history = []
    val_loss_history = []
    ssim_loss_history = []
    psnr_loss_history = []
    
    for epoch in range(opts.epoch_num):
        print("Lr_G:{}".format(optimizer_G.state_dict()['param_groups'][0]['lr']))
        print("Lr_D:{}".format(optimizer_D.state_dict()['param_groups'][0]['lr']))
        t0 = time.perf_counter()
        model_G.train()
        model_D.train()
        
        train_g_loss = 0
        train_d_loss = 0
        train_sample_num = 0

        
        # training for one epoch
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{opts.epoch_num}', unit='img') as pbar:
            for batch_id, (imgs, gt_mask,mask, _) in enumerate(train_loader):
                n = len(imgs)
                imgs = imgs.cuda()
                gt_mask = gt_mask.cuda()
                ## Train Discriminator
                optimizer_D.zero_grad()
                with torch.no_grad():
                    fake = model_G(imgs)
                pred_fake = model_D(torch.cat([imgs, fake.detach()], 1))
                pred_real = model_D(torch.cat([imgs, gt_mask], 1))
                real_label = torch.FloatTensor(pred_real.data.size()).fill_(1.0).cuda()
                fake_label = torch.FloatTensor(pred_fake.data.size()).fill_(0.0).cuda()
                loss_D = 0.5*D_loss(pred_fake,fake_label)+ 0.5*D_loss(pred_real, real_label)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                ##  Train Generators
                optimizer_G.zero_grad()
                output = model_G(imgs)
                pred_fake = model_D(torch.cat([imgs, output], 1))
                # Adversarial loss
                
                loss_GAN = D_loss(pred_fake,torch.FloatTensor(pred_fake.data.size()).fill_(1.0).cuda())
                loss_l1 = criterion(gt_mask, output)
                loss = loss_l1+opts.adv_loss_weight*loss_GAN.clamp(0,1)
                loss.backward()
                optimizer_G.step()
                train_g_loss += n * loss.item()
                train_d_loss += n * loss_D.item()
                train_sample_num += n
                pbar.set_postfix(**{'loss_G (batch)': loss.item(), 'loss_D (batch)': loss_D.item()})
                pbar.update(imgs.shape[0])
                if batch_id > 112:
                    break
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        train_g_loss = train_g_loss / train_sample_num
        train_d_loss = train_d_loss / train_sample_num
        print('Epoch {0:d}/{1:d}---Finished'.format(epoch + 1, opts.epoch_num))
        print('Training loss: G_loss: {:.6f},D_loss: {:.6f}'.format(train_g_loss, train_d_loss))
        train_g_loss_history.append(train_g_loss)
        train_d_loss_history.append(train_d_loss)
        # validation
        # val_save_dir = result_path + '/val'
        # if not os.path.exists(val_save_dir):
        #     os.makedirs(val_save_dir)
        # ssim, psnr = 0.0, 0.0
        # if (epoch % opts.val_intel == 0):
        #     val_loss, ssim, psnr = eval_net(opts, outdir=val_save_dir, net=model_G, loader=val_loader
        #                                     , epoch=epoch, criterion=mertics,)
        #
        # print('Valing loss: {:.6f},ssim: {:.6f},psnr loss: {:.6f}'.format(val_loss, ssim, psnr))
        #
        writer.add_scalar('train_g_loss', train_g_loss, epoch)
        writer.add_scalar('train_d_loss', train_d_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('ssim', ssim, epoch)
        # writer.add_scalar('psnr', psnr, epoch)
        val_loss_history.append(val_loss)
        # ssim_loss_history.append(ssim)
        # psnr_loss_history.append(psnr)
        # save best model
        if epoch+1==opts.epoch_num:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model_G.state_dict(),
                    "epoch": epoch,
                    "epoch_loss": best_val_loss,
                },
                os.path.join(result_path, f"model_best_loss.pth"),
            )
            torch.save(
                {
                    "model_state_dict": model_D.state_dict(),
                    "epoch": epoch,
                    "epoch_loss": best_val_loss,
                },
                os.path.join(result_path, f"discriminator_model_best_loss.pth"),
            )
        epochs = range(1, len(train_g_loss_history) + 1)
        plt.ylim(ymin=0, ymax=1)
        plt.plot(epochs, train_g_loss_history, 'y', label='Training G loss')
        # plt.plot(epochs, train_d_loss_history, 'b', label='Training D loss')
        # plt.plot(epochs, val_loss_history, 'r', label='Validation loss')
        # plt.plot(epochs, ssim_loss_history, 'r', label='SSIM')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.legend()
        plt.savefig(result_path + '/loss.png')
    
    plt.close()
    
    epochs = range(1, len(train_g_loss_history) + 1)
    df = pd.DataFrame({'epochs': epochs, 'G loss': train_g_loss_history,
                       'D loss': train_d_loss_history, })
    save_path = os.path.join(result_path, f'loss.csv')
    df.to_csv(path_or_buf=save_path, sep=',', na_rep='NA',
              columns=['epochs', 'G loss', 'D loss'])
    # predict(opts, val_path, result_path, model_G)


def write_opts(opts, sava_path):
    argsDict = opts.__dict__
    with open(sava_path + r'/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def cgan_main(user_name,fluorescence,weight_name):
    
    parser = TrainOptions()
    opts = parser.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dataroot = opts.dataroot
    save_dir = opts.save_dir
    pb_aug = opts.pb_aug
    model_name = 'cgan'
    cell_and_magnification =user_name
    fluorescence = fluorescence
    weights_path = fr'./model_weight/cgan_{fluorescence}/{weight_name}'
    opts.weights_path = weights_path
    print(opts.weights_path)
    opts.flu = fluorescence
    print(opts.flu)
    
    train_path = os.path.join(dataroot, cell_and_magnification, 'pbda')
    # val_path = os.path.join(dataroot, cell_and_magnification, 'val_and_test')
 
  
    result_path = os.path.join(save_dir, cell_and_magnification,
                               fluorescence, model_name, pb_aug, f'result0')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    write_opts(opts, result_path)
    model_G = Unet_drop(in_ch=opts.inch)
    model_D = Discriminator(input_nc=opts.inch+1)
    model_G.cuda()
    model_D.cuda()
    train(opts, train_path, result_path, model_G=model_G, model_D=model_D)

if __name__ == "__main__":
    parser = TrainOptions()
    opts = parser.parse()
    
