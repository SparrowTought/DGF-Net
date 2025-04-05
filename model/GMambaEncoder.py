import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
import math
from .GMamba import GASSM
from einops import rearrange, repeat
import matplotlib.pyplot as plt


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dwconv = DWConv(in_features)
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class Block(nn.Module):

    def __init__(self, dim,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 depths=0,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv: int = 3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.0,
                 ssm_init="v0",
                 forward_type="v2",
                 id_stage=0,
                 nums=0,
                 HW = 0,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.depth = depths
        self.drop_path = drop_path
        # self.short = Short(dim, dim)
        self.id_stage = id_stage
        self.num = nums
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0

        if self.ssm_branch:
            self.norm = norm_layer(dim // 2)
            self.op = GASSM(
                d_model=dim ,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FDFG(dim)
 
        self.gelu = nn.GELU()
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



    def forward(self, x, mask):
        ox = self.op(self.norm1(x),mask)
        att_outputs =  self.drop_path(ox) + x
        att_outputs = att_outputs + self.drop_path(self.mlp(self.norm2(att_outputs)))
        return att_outputs

class stdmean(nn.Module):
    def __init__(self,):
        super(stdmean, self).__init__()
        self.sig =  nn.Sigmoid()

        self.num_images = 0

        self.eps = 1e-10
    def forward(self, x1):

        std = torch.std(x1, dim=1, keepdim=True)

        mean1 = torch.mean(x1, dim=1, keepdim=True)
        filtered_s1 = torch.where(x1 > mean1, x1, torch.zeros_like(x1))
        nonzero_count_s1 = torch.where(filtered_s1 > 0, torch.ones_like(filtered_s1), torch.zeros_like(filtered_s1)).sum(
            dim=1, keepdim=True).float()
        nonzero_count_s1 = torch.where(nonzero_count_s1 == 0, torch.ones_like(nonzero_count_s1), nonzero_count_s1)
        nonzero_count_s1 = filtered_s1.sum(dim=1, keepdim=True) / nonzero_count_s1
        x1 = (x1 - nonzero_count_s1) / (std + self.eps)

        sig_x1 = self.sig(x1)

        x1 = sig_x1 * x1 + x1

        return x1
class Frequency(nn.Module):
    def __init__(self, nc):
        super(Frequency, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc // 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            stdmean(),
            nn.Conv2d(nc // 2, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc // 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            stdmean(),
            nn.Conv2d(nc // 2, nc, 1, 1, 0))

    def forward(self, x):
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out



class FDFG(nn.Module):

    def __init__(self, in_nc):
        super(FDFG, self).__init__()
        self.glu = nn.Sequential(nn.Conv2d(in_nc, in_nc, 1),
                                             nn.Sigmoid())
        self.frequency = Frequency(in_nc)

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_freq = self.frequency(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_out = x * self.glu(x_freq_spatial)
        x_out = x_out.permute(0, 2, 3, 1)

        return x_out
class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class GradientAwareMamba(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512, 1024],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4,4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
            assert patch_size == 4


            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
                (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()))


        def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
            # return PatchMerging2D(dim, out_dim)
            return nn.Sequential(

                Permute(0, 3, 1, 2),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1),
                Permute(0, 2, 3, 1),
                norm_layer(out_dim),
            )

        self.pool1 = nn.MaxPool2d(2, 2)
        self.mask_conv1 = nn.Conv2d(1, 1, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.mask_conv2 = nn.Conv2d(1, 1, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.mask_conv3 = nn.Conv2d(1, 1, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.mask_conv4 = nn.Conv2d(1, 1, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.mask_conv5 = nn.Conv2d(1, 1, 3, 1, 1)
        # patch_embed
        self.patch_embed1 = _make_patch_embed_v2(in_chans=1, embed_dim=embed_dims[0], patch_size=4,
                                                 norm_layer=nn.LayerNorm
                                                 )
        self.patch_embed2 = _make_downsample_v3(dim=embed_dims[0],
                                               out_dim=embed_dims[1], norm_layer=nn.LayerNorm)
        self.patch_embed3 = _make_downsample_v3(dim=embed_dims[1],
                                               out_dim=embed_dims[2], norm_layer=nn.LayerNorm)
        self.patch_embed4 = _make_downsample_v3(dim=embed_dims[2],
                                               out_dim=embed_dims[3], norm_layer=nn.LayerNorm)
        self.patch_embed5 = _make_downsample_v3(dim=embed_dims[3],
                                               out_dim=embed_dims[4], norm_layer=nn.LayerNorm)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], mlp_ratio=mlp_ratios[0],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", id_stage=i, nums=0, HW= 256 // 4 * 256 // 4)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], mlp_ratio=mlp_ratios[1],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", id_stage=i, nums=0, HW= 256 // 8 * 256 // 8)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], mlp_ratio=mlp_ratios[2],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", id_stage=i, nums=2, HW= 256 // 16 * 256 // 16)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], mlp_ratio=mlp_ratios[3],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", id_stage=i, nums=2, HW= 256 // 32 * 256 // 32)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        cur += depths[3]
        self.block5 = nn.ModuleList([Block(
            dim=embed_dims[4], mlp_ratio=mlp_ratios[4],
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3, ssm_conv_bias=True,
            ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2", id_stage=i, nums=2, HW= 256 // 32 * 256 // 32)
            for i in range(depths[4])])
        self.norm5 = norm_layer(embed_dims[4])
        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[3]
        for i in range(self.depths[4]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x1):
        B = x1.shape[0]
        mask1  = x1
        outs1 = []

  
        # stage 1
        x1 = self.patch_embed1(x1)


        mask1 = self.mask_conv1(mask1)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1,mask1)
        x1 = self.norm1(x1)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)


        # stage 2
        x1 = self.patch_embed2(x1)
        mask1 = self.pool2(mask1)
        mask1 = self.mask_conv2(mask1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, mask1)
        x1 = self.norm2(x1)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)


        # stage 3

        x1 = self.patch_embed3(x1)

        mask1 = self.pool3(mask1)
        mask1 = self.mask_conv3(mask1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, mask1)
        x1 = self.norm3(x1)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)


        x1 = self.patch_embed4(x1)
        mask1 = self.pool4(mask1)
        mask1 = self.mask_conv4(mask1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1,mask1)
        x1 = self.norm4(x1)
        x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        outs1.append(x1_1)
     
        # x1 = self.patch_embed5(x1)
        # mask1 = self.pool5(mask1)
        # mask1 = self.mask_conv5(mask1) + mask1
        # for i, blk in enumerate(self.block5):
        #     x1 = blk(x1,mask1)
        # x1 = self.norm5(x1)
        # x1_1 = rearrange(x1, "b h w c -> b c h w").contiguous()
        # outs1.append(x1_1)

        return outs1


    def forward(self, x1):
        x1 = self.forward_features(x1)
        # x = self.head(x)

        return x1


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                      groups=dim)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # BCHW
        x = self.conv(x)
        out = x.permute(0, 3, 2, 1)  # BHWC

        return out


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict





@register_model
class GMEncoder(GradientAwareMamba):
    def __init__(self, **kwargs):
        super(GMEncoder, self).__init__(
            patch_size=4, embed_dims=[16, 32, 64, 128, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1, 1, 1, 1], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

