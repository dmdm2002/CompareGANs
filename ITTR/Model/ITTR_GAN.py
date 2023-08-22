import torch
import torch.nn as nn
import torch.nn.functional as F
from ITTR_pytorch import HPB

import torchsummary
import numpy as np


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class ITTRModel(nn.Module):
    def __init__(self):
        super(ITTRModel, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down_conv_1 = self._conv_layer(3, 64, kernel_size=7, stride=1, padding=0)
        self.down_sampling_1 = Downsample(64, pad_type='reflect', filt_size=7, stride=1)

        self.down_conv_2 = self._conv_layer(64, 128, kernel_size=3, stride=2, padding=1)
        self.down_sampling_2 = Downsample(128, pad_type='reflect', filt_size=3, stride=2)

        self.down_conv_3 = self._conv_layer(128, 256, kernel_size=3, stride=2, padding=1)
        self.down_sampling_3 = Downsample(256, pad_type='reflect', filt_size=3, stride=2)

        self.hpb_blocks = self._hpb_block(9)

        self.up_conv_1 = self._conv_layer(256, 128, kernel_size=3, stride=2, padding=1)
        self.up_conv_2 = self._conv_layer(128, 64, kernel_size=3, stride=2, padding=1)
        self.up_conv_3 = self._conv_layer(64, 3, kernel_size=7, stride=1, padding=1)

    def _conv_layer(self, in_dims, out_dims, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_dims),
            nn.GELU(),
        )

    def _hpb_block(self, repeat):
        attns = []
        for i in range(repeat):
            attns.append(HPB(dim=512, dim_head=32, heads=8, attn_height_top_k=16, attn_width_top_k=16,
                             attn_dropout=0., ff_mult=4, ff_dropout=0.)
                         )

        return nn.Sequential(*attns)

    def forward(self, x):
        x = self.down_conv_1(x)
        x = self.down_sampling_1(x)
        print(x.shape)
        x = self.down_conv_2(x)
        x = self.down_sampling_2(x)
        print(x.shape)
        x = self.down_conv_3(x)
        x = self.down_sampling_3(x)
        print(x.shape)
        # attn = self.hpb_blocks(x)

        x = self.up(x)
        x = self.up_conv_1(x)

        x = self.up(x)
        x = self.up_conv_2(x)
        print(x.shape)
        x = self.up_conv_3(x)
        print(x.shape)

        out = nn.functional.tanh(x)

        return out


a = ITTRModel()
torchsummary.summary(a, (3, 224, 224), device='cpu')