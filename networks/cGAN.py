
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class DoubleConv(nn.Module):
    ''' (Conv -> BN -> Relu) -> (Conv -> BN -> Relu) '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# class DoubleConv(nn.Module):
#    ''' (Conv -> BN -> Relu) -> (Conv -> BN -> Relu) '''
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.double_conv = nn.Sequential(
#                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                nn.InstanceNorm2d(out_channels),
#                nn.ReLU(inplace = True),
#                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                nn.InstanceNorm2d(out_channels),
#                nn.ReLU(inplace=True),
#                )
#    def forward(self, x):
#        return self.double_conv(x)
#

class Encode(nn.Module):
    '''Encode : Downscaling with maxpooling then double conv'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block_conv(x)
    
class Decode(nn.Module):
    '''Decode : Upscaling with linear or Conv method'''

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.decode = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.decode = nn.ConvTranspose2d(in_channels // 2,
                                             in_channels // 2,
                                             kernel_size=2,
                                             stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.decode(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # corping
        #        x2 = x2[:, diffY // 2 : diffY - diffY // 2, diffX // 2 : diffX - diffX // 2 ]
        # padding
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    '''Output conv layer with 1*1 conv'''

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        
        x_tanh_scaled=self.conv(x)
        # if normrailze ,use torch.tanh
        # x_tanh =torch.tanh(self.conv(x))
        # x_tanh_scaled = (1 + x_tanh) / 2
        return x_tanh_scaled
    
    
class Unet(nn.Module):

    def __init__(self, in_ch=1, n_classes=1, bilinear=True):
        super(Unet, self).__init__()
        self.n_chnnels = in_ch
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.in_fch = 32

        self.inc = DoubleConv(in_ch, self.in_fch)
        self.encode1 = Encode(self.in_fch, self.in_fch*2)
        self.encode2 = Encode(self.in_fch*2, self.in_fch*4)
        self.encode3 = Encode(self.in_fch*4, self.in_fch*8)
        self.encode4 = Encode(self.in_fch*8, self.in_fch*8)
        self.decode1 = Decode(self.in_fch*16, self.in_fch*4, bilinear)
        self.decode2 = Decode(self.in_fch*8, self.in_fch*2, bilinear)
        self.decode3 = Decode(self.in_fch*4, self.in_fch, bilinear)
        self.decode4 = Decode(self.in_fch*2, self.in_fch, bilinear)
        self.out_conv = OutConv(self.in_fch, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encode1(x1)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.encode4(x4)
        x = self.decode1(x5, x4)
        x = self.decode2(x, x3)
        x = self.decode3(x, x2)
        x = self.decode4(x, x1)
        logits = self.out_conv(x)
        return logits




class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):

        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)