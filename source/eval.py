import torch

from tqdm import tqdm

from skimage.metrics import structural_similarity as SSIM
from source.utils import sample_image,sample_image,PSNR


def eval_net(opts,outdir, net, loader, epoch, criterion):
  
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    train_loss = 0
    ssim_loss = 0
    psnr_loss = 0
    train_sample_num = 0
    
    # with tqdm(total=n_val, desc='Validation', unit='batch',leave=False) as pbar:
    for it, (img, gt_mask,_,_) in enumerate(loader):
        n = len(img)
        img = img.cuda()
        gt_mask = gt_mask.cuda()
        with torch.no_grad():
         
           prob = net(img)
        
        loss = criterion(prob, gt_mask)
        ssim = SSIM(prob[0, 0, :, :].cpu().numpy(), gt_mask[0, 0, :, :].cpu().numpy())
        psnr = PSNR(prob[0, 0, :, :].cpu().numpy(), gt_mask[0, 0, :, :].cpu().numpy())
     
        train_loss += n * loss.item()
        ssim_loss += n * ssim
        psnr_loss += n * psnr
        train_sample_num += n
    
        if it == 1:
            sample_image(opts,outdir, epoch, gt_mask, prob)
        # pbar.update()
    net.train()
    return train_loss / train_sample_num,ssim_loss/train_sample_num,psnr_loss/train_sample_num