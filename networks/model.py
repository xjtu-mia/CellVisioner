import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
from torch.autograd import Variable


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(double_conv, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        if self.dropout > 0.01:
            x = nn.Dropout(p=self.dropout)(x)
        y = self.conv2(x)
        return y


# Encoding block in U-Net
class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dropout)
        self.down = nn.MaxPool2d(2)
    
    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)
        return y, y_conv


# Decoding block in U-Net
class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, bilinear=True):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dropout)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
    
    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)
        return y, y_conv


def concatenate(x1, x2):
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2))
    y = torch.cat([x2, x1], dim=1)
    return y


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return x


class outtanh(nn.Module):
    '''Output conv layer with 1*1 conv'''
    
    def __init__(self):
        super(outtanh, self).__init__()
        self.tanh = torch.tanh
    
    def forward(self, x):
        x_tanh = self.tanh(x)
        x_tanh_scaled = (1 + x_tanh) / 2
        return x_tanh_scaled


class outsoftmax(nn.Module):
    def __init__(self):
        super(outsoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.softmax(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, bilinear=False):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.in_fch = 32
        
        self.enc1 = enc_block(in_ch, self.in_fch, dropout=0.1)
        self.enc2 = enc_block(self.in_fch, self.in_fch * 2, dropout=0.1)
        self.enc3 = enc_block(self.in_fch * 2, self.in_fch * 4, dropout=0.2)
        self.enc4 = enc_block(self.in_fch * 4, self.in_fch * 8, dropout=0.2)
        self.dec1 = dec_block(self.in_fch * 8, self.in_fch * 8, dropout=0.3, bilinear=bilinear)
        self.dec2 = dec_block(self.in_fch * 16, self.in_fch * 4, dropout=0.2, bilinear=bilinear)
        self.dec3 = dec_block(self.in_fch * 8, self.in_fch * 2, dropout=0.2, bilinear=bilinear)
        self.dec4 = dec_block(self.in_fch * 4, self.in_fch, dropout=0.1, bilinear=bilinear)
        self.outconv = double_conv(self.in_fch * 2, self.in_fch, dropout=0.1)
        
        self.pb_outc = outconv(self.in_fch, 1)
        # self.pb_outs = outtanh()
    
    def forward(self, x):
        enc1, enc1_conv = self.enc1(x)
        enc2, enc2_conv = self.enc2(enc1)
        enc3, enc3_conv = self.enc3(enc2)
        enc4, enc4_conv = self.enc4(enc3)
        dec1, dec1_conv = self.dec1(enc4)
        dec2, dec2_conv = self.dec2(concatenate(dec1, enc4_conv))
        dec3, dec3_conv = self.dec3(concatenate(dec2, enc3_conv))
        dec4, dec4_conv = self.dec4(concatenate(dec3, enc2_conv))
        dec_out = self.outconv(concatenate(dec4, enc1_conv))
        
        pb = self.pb_outc(dec_out)
        # pb = self.pb_outs(pb)
        return pb
    
    def name(self):
        return 'Network (Input channel = {0:d})'.format(self.in_ch)