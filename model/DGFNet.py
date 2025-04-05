from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile
from .GMambaEncoder import GMEncoder

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Channel(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ca = Channel(F_g=out_channels, F_x=out_channels)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.conv(x)
        skip_x_att = self.ca(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)



class DGF(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, mode='train'):
        super().__init__()

        self.backbone = GMEncoder()
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 16

        self.decoder4 = Decoder(in_channels * 16, in_channels * 8, nb_Conv=2)
        self.decoder3 = Decoder(in_channels * 8, in_channels * 4, nb_Conv=2)
        self.decoder2 = Decoder(in_channels * 4, in_channels * 2, nb_Conv=2)
        self.decoder1 = Decoder(in_channels * 2, in_channels, nb_Conv=2)
        self.out = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))


        self.loss5 = nn.Sequential(nn.Conv2d(in_channels * 16, 1, 1))
        self.loss4 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
        self.loss3 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
        self.loss2 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
        self.outconv = nn.Conv2d(4 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        backbone = self.backbone(x)
        x1, x2, x3, x4 = backbone[0], backbone[1], backbone[2], backbone[3]

        d3 = self.decoder3(x4, x3)
        d2 = self.decoder2(d3, x2)
        out = self.out(self.up_decoder1(d2, x1))

        gt_4 = self.loss4(x4)
        gt_3 = self.loss3(d3)
        gt_2 = self.loss2(d2)
        gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
        gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
        gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
        d0 = self.outconv(torch.cat((gt2, gt3, gt4, out), 1))

        if self.mode == 'train':
            return (torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
        else:
            return torch.sigmoid(out)

