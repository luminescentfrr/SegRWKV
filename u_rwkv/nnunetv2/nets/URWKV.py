from typing import Sequence
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from os.path import join as pjoin
from collections import OrderedDict
from mmengine.model.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from monai.networks.blocks.dynunet_block import UnetOutBlock
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from torch.utils.cpp_extension import load
wkv_cuda = load(name="bi_wkv", sources=["/home/user/公共的/U-RWKV/u_rwkv/nnunetv2/nets/cuda_new/bi_wkv.cpp", "/home/user/公共的/U-RWKV/u_rwkv/nnunetv2/nets/cuda_new/bi_wkv_kernel.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_86,code=sm_86'])

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False,
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, init_mode,
                                   key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, hidden_rate,
                                   init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

@BACKBONES.register_module()
class VRWKV(BaseModule):
    """
    norm w and norm u
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=256,
                 depth=12,
                 drop_path_rate=0.,
                 channel_gamma=1/4,
                 shift_pixel=1,
                 init_values=None,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 pretrained=None,
                 with_cp=False,
                 init_cfg=None):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        super().__init__(self.init_cfg)
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate
        logger = get_root_logger()
        logger.info(f'layer_scale: {init_values is not None}')

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                init_values=init_values,
                post_norm=post_norm,
                key_norm=key_norm,
                with_cp=with_cp
            ))


        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)
    

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)

                out = patch_token
                outs.append(out)

        return tuple(outs)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    @torch.no_grad()
    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    支持非正方形输入，所有空间尺寸推断和特征对齐均分别处理高和宽。
    """

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_h, in_w = x.size()  # 支持非正方形输入，分别记录高和宽
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            # 分别计算期望的高和宽
            right_h = int(in_h / 4 / (i+1))
            right_w = int(in_w / 4 / (i+1))
            # 若实际输出尺寸与期望不符，则进行padding/cropping
            if x.size()[2] != right_h or x.size()[3] != right_w:
                pad_h = right_h - x.size()[2]
                pad_w = right_w - x.size()[3]
                # 断言padding/cropping量在合理范围内
                assert abs(pad_h) < 3 and abs(pad_h) >= 0, f"x {x.size()} should {right_h} (h)"
                assert abs(pad_w) < 3 and abs(pad_w) >= 0, f"x {x.size()} should {right_w} (w)"
                # 新建目标特征张量
                feat = torch.zeros((b, x.size()[1], right_h, right_w), device=x.device)
                # 计算实际可拷贝区域
                copy_h = min(x.size()[2], right_h)
                copy_w = min(x.size()[3], right_w)
                # 将x拷贝到feat左上角
                feat[:, :, 0:copy_h, 0:copy_w] = x[:, :, 0:copy_h, 0:copy_w]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
    



class ModulatedConv2d(nn.Module):
    """
    Modulated Convolution with residual connection and spatial modulation.
    
    Structure: Conv2d(expand) → SpatialModulation → Residual → BatchNorm → Conv1x1(compress)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        padding: Padding added to input (default: 0)
        stride: Stride of the convolution (default: 1)
        growth_rate: Channel expansion ratio for intermediate features (default: 2.0)
        use_batchnorm: Whether to use batch normalization (default: True)
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            growth_rate=2.0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        # Main convolution with channel expansion
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not use_batchnorm
        )
        self.relu = nn.ReLU(inplace=True)
        # Batch normalization (optional)
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        


    
    def forward(self, x):
        # Main convolution
        x = self.conv(x)
        
        # Spatial modulation
        x_pool = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x_weight = F.softmax(x_pool, dim=1)         # Softmax normalization
        x_modulated = x * x_weight                  # Element-wise weighting
        
        # Residual connection
        x = x + x_modulated

        x = self.relu(x)
        
        # Batch normalization
        if self.bn is not None:
            x = self.bn(x)
        
        return x
   



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            growth_rate=2.0,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = ModulatedConv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = ModulatedConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, skip_channels, n_skip=3, deep_supervision=False):
        super().__init__()
        self.n_skip = n_skip
        self.deep_supervision = deep_supervision
        
        # 特征转换层
        self.conv_more = Conv2dReLU(
            encoder_channels,
            decoder_channels[0],
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        # 解码器块
        in_channels = [decoder_channels[0]] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        # 根据 n_skip 调整跳跃连接通道
        if self.n_skip != 0:
            self.skip_channels = list(skip_channels)
            for i in range(4-self.n_skip):
                self.skip_channels[3-i] = 0
        else:
            self.skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) 
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, self.skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        """
        支持非正方形patch网格：
        - 若hidden_states为patch序列，优先从features[0]获取patch网格高宽
        - 若无features，回退为正方形假设
        - 所有空间尺寸推断均兼容非正方形输入
        """
        if len(hidden_states.shape) == 4:
            x = hidden_states
        else:
            B, n_patch, hidden = hidden_states.size()
            # 支持非正方形patch网格
            # features[0]的空间尺寸即为patch网格的高宽
            if features is not None and len(features) > 0:
                h, w = features[0].shape[2], features[0].shape[3]
            else:
                # 回退到正方形假设
                h = w = int(np.sqrt(n_patch))
            assert h * w == n_patch, f"patch数量{n_patch}与特征图尺寸{h}x{w}不符"
            x = hidden_states.permute(0, 2, 1)
            x = x.contiguous().view(B, hidden, h, w)
        
        x = self.conv_more(x)
        
        # 收集深度监督特征
        decoder_outputs = []
        
        # 逐步解码
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            
            # 如果启用深度监督，收集每层输出
            if self.deep_supervision:
                decoder_outputs.append(x)
        
        if self.deep_supervision:
            return x, decoder_outputs
        else:
            return x


class URWKVEncoder(BaseModule):
    """
    URWKVEncoder: Unified encoder combining ResNet and VRWKV
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=256,
                 depth=12,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 channel_gamma=1/4,
                 shift_pixel=1,
                 init_values=None,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 with_cp=False):
        super().__init__()
        
        # ResNet backbone for feature extraction
        self.resnet = ResNetV2(block_units=[3, 3, 3], width_factor=1.0)
        
        # 动态计算VRWKV的img_size参数
        # ResNet总降采样倍数：stride=2(root) * stride=2(maxpool) * stride=2(block2) * stride=2(block3) = 16x
        resnet_downsample_ratio = 16
        vrwkv_img_size = img_size // resnet_downsample_ratio
        vrwkv_patch_size = max(1, patch_size // resnet_downsample_ratio)  # 确保patch_size至少为1
        
        # VRWKV encoder
        self.vrwkv = VRWKV(
            img_size=vrwkv_img_size,  # 动态计算的图像尺寸
            patch_size=vrwkv_patch_size,  # 动态计算的patch尺寸
            in_channels=self.resnet.width * 16,  # ResNet output channels
            embed_dims=embed_dims,
            depth=depth,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            init_values=init_values,
            shift_mode=shift_mode,
            init_mode=init_mode,
            post_norm=post_norm,
            key_norm=key_norm,
            hidden_rate=hidden_rate,
            final_norm=final_norm,
            interpolate_mode=interpolate_mode,
            with_cp=with_cp
        )
    
    def forward(self, x):
        # 处理通道数不匹配的问题 - 医学图像通常是单通道，但ResNet期望3通道输入
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # 将1通道扩展为3通道
        
        # ResNet feature extraction
        resnet_out, resnet_features = self.resnet(x)
        
        # VRWKV processing
        vrwkv_out = self.vrwkv(resnet_out)
        if isinstance(vrwkv_out, tuple):
            vrwkv_out = vrwkv_out[0]
        
        return vrwkv_out, resnet_features


@BACKBONES.register_module()
class U_RWKV(BaseModule):
    """
    URWKV: U-Net with RWKV backbone
    Combines ResNet feature extraction with VRWKV processing and U-Net decoder
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_channels=1,
                 embed_dims=768,
                 depth=12,
                 decoder_channels=(256, 128, 64, 32),
                 skip_channels=(512, 256, 64, 0),
                 n_skip=3,
                 deep_supervision=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 channel_gamma=1/4,
                 shift_pixel=1,
                 init_values=None,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 pretrained=None,
                 with_cp=False,
                 init_cfg=None):
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        super().__init__(self.init_cfg)
        
        self.out_channels = out_channels
        self.n_skip = n_skip
        self.deep_supervision = deep_supervision
        
        # Unified encoder
        self.encoder = URWKVEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            depth=depth,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            init_values=init_values,
            shift_mode=shift_mode,
            init_mode=init_mode,
            post_norm=post_norm,
            key_norm=key_norm,
            hidden_rate=hidden_rate,
            final_norm=final_norm,
            interpolate_mode=interpolate_mode,
            with_cp=with_cp
        )
        
        # Decoder
        self.decoder = DecoderCup(
            encoder_channels=embed_dims,
            decoder_channels=decoder_channels,
            skip_channels=skip_channels,
            n_skip=n_skip,
            deep_supervision=deep_supervision
        )
        
        # Segmentation heads using UnetOutBlock
        if deep_supervision:
            # 为每个解码层创建输出头
            self.out_layers = nn.ModuleList([
                UnetOutBlock(
                    spatial_dims=2,
                    in_channels=decoder_channels[i],
                    out_channels=out_channels
                )
                for i in range(len(decoder_channels))
            ])
        else:
            # 单个输出头
            self.out_layers = nn.ModuleList([
                UnetOutBlock(
                    spatial_dims=2,
                    in_channels=decoder_channels[-1],
                    out_channels=out_channels
                )
            ])

    def forward(self, x):
        # Unified encoder processing
        encoder_out, skip_features = self.encoder(x)
        
        # Decoder with skip connections
        if self.deep_supervision:
            decoder_out, decoder_features = self.decoder(encoder_out, skip_features)
            
            # 深度监督输出：使用decoder_features的4个输出，对应尺度从粗到细
            # decoder_features[0]: 40×40, 256通道 → 0.125尺度
            # decoder_features[1]: 80×80, 128通道 → 0.25尺度  
            # decoder_features[2]: 160×160, 64通道 → 0.5尺度
            # decoder_features[3]: 320×320, 32通道 → 1.0尺度
            out = []
            for i, feat in enumerate(decoder_features):
                pred = self.out_layers[i](feat)
                out.append(pred)
            
            return out  # 返回4个输出，对应深度监督的4个尺度
        else:
            decoder_out = self.decoder(encoder_out, skip_features)
            # 单一输出
            out = self.out_layers[-1](decoder_out)
            return out
    
    @torch.no_grad()
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def load_from(self, resnet_weights):
        """
        加载ResNet权重，用于研究VRWKV预训练权重的影响
        
        Args:
            resnet_weights: numpy数组字典，包含ResNet的预训练权重
        """
        from mmseg.utils import get_root_logger
        logger = get_root_logger()
        
        logger.info("Loading ResNet weights using internal load_from method")
        
        # 定义ResNet结构配置
        block_units = [3, 4, 9]  # 匹配权重文件结构
        block_names = ['block1', 'block2', 'block3']
        
        # 加载root层权重
        try:
            # conv_root
            conv_root_weight = np2th(resnet_weights['conv_root/kernel'], conv=True)
            self.encoder.resnet.root.conv.weight.copy_(conv_root_weight)
            
            # gn_root
            gn_root_weight = np2th(resnet_weights['gn_root/scale'])
            gn_root_bias = np2th(resnet_weights['gn_root/bias'])
            self.encoder.resnet.root.gn.weight.copy_(gn_root_weight.view(-1))
            self.encoder.resnet.root.gn.bias.copy_(gn_root_bias.view(-1))
            
            logger.info("Root layer weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load root layer weights: {str(e)}")
            return False
        
        # 加载各个block的权重
        for block_idx, (block_name, num_units) in enumerate(zip(block_names, block_units)):
            try:
                block_module = getattr(self.encoder.resnet.body, block_name)
                
                for unit_idx in range(1, num_units + 1):
                    unit_name = f"unit{unit_idx}"
                    unit_module = getattr(block_module, unit_name)
                    
                    # 调用PreActBottleneck的load_from方法
                    unit_module.load_from(resnet_weights, block_name, unit_name)
                
                logger.info(f"{block_name} weights loaded successfully ({num_units} units)")
                
            except Exception as e:
                logger.error(f"Failed to load {block_name} weights: {str(e)}")
                return False
        
        logger.info("All ResNet weights loaded successfully")
        return True


def load_pretrained_ckpt(model, 
                         vrwkv_path="/home/user/公共的/U-RWKV/data/pretrained/vrwkv_b_in1k_224.pth", 
                         strict=False):
    """
    简化版权重加载函数，只处理VRWKV权重
    ResNet权重请使用PreActBottleneck.load_from()方法直接加载

    Args:
        model: U_RWKV模型实例
        vrwkv_path: VRWKV权重文件路径
        strict: 是否严格匹配权重键名

    Returns:
        model: 加载权重后的模型实例
    """
    logger = get_root_logger()
    
    # 验证模型实例
    if not isinstance(model, U_RWKV):
        raise TypeError(f"Expected U_RWKV model, got {type(model)}")
    
    loading_status = {
        'vrwkv_loaded': False,
        'errors': []
    }
    
    # 加载 VRWKV 权重
    if vrwkv_path is not None:
        try:
            import os
            if not os.path.exists(vrwkv_path):
                error_msg = f"VRWKV weights file not found: {vrwkv_path}"
                loading_status['errors'].append(error_msg)
                logger.error(error_msg)
                return loading_status
            
            logger.info(f"Loading VRWKV weights from: {vrwkv_path}")
            
            # 加载权重字典
            checkpoint = torch.load(vrwkv_path, map_location='cpu', weights_only=False)
            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
            
            # 需要跳过的权重键名
            skip_keys = [
                'head.fc.weight',
                'head.fc.bias',
                'backbone.ln1.weight',
                'backbone.ln1.bias',
                'backbone.patch_embed.projection.weight',
                'backbone.patch_embed.projection.bias',
                'backbone.pos_embed'
            ]
            
            # 过滤权重字典
            filtered_dict = {}
            skipped_keys = []
            
            for key, value in pretrained_dict.items():
                if any(skip_key in key for skip_key in skip_keys):
                    skipped_keys.append(key)
                    continue
                filtered_dict[key] = value
            
            logger.info(f"Skipped {len(skipped_keys)} incompatible keys: {skipped_keys}")
            
            # 键名映射：backbone.* → encoder.vrwkv.*
            mapped_dict = {}
            for key, value in filtered_dict.items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', 'encoder.vrwkv.')
                    mapped_dict[new_key] = value
                else:
                    mapped_dict[key] = value
            
            logger.info(f"Mapped {len(mapped_dict)} keys for VRWKV model")
            
            # 加载到模型
            missing_keys, unexpected_keys = model.load_state_dict(mapped_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys in VRWKV loading: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in VRWKV loading: {unexpected_keys}")
            
            loading_status['vrwkv_loaded'] = True
            logger.info("VRWKV weights loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load VRWKV weights: {str(e)}"
            loading_status['errors'].append(error_msg)
            logger.error(error_msg)
    
    # 更新模型权重并返回模型
    model_dict = model.state_dict()
    model.load_state_dict(model_dict)
    
    return model


def get_u_rwkv_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = False,
    use_pretrain: bool = True,
    resnet_path: str = "/home/user/公共的/U-RWKV/data/pretrained/R50.npz"
):

    
    label_manager = plans_manager.get_label_manager(dataset_json)

    model = U_RWKV(
        in_channels=num_input_channels,
        out_channels=label_manager.num_segmentation_heads,
        embed_dims=768,
        decoder_channels=(256, 128, 64, 32),
        skip_channels=(512, 256, 64, 0),
        deep_supervision=deep_supervision,
        img_size=224,
        patch_size=16,
        depth=12,
        init_values=1e-6  # 添加这一行来启用layer scale
    )

    # 加载ResNet权重
    if resnet_path is not None:
        import numpy as np
        import os
        from mmseg.utils import get_root_logger
        
        logger = get_root_logger()
        
        if os.path.exists(resnet_path):
            logger.info(f"Loading ResNet weights from: {resnet_path}")
            resnet_weights = np.load(resnet_path)
            success = model.load_from(resnet_weights)
            if success:
                logger.info("ResNet weights loaded successfully")
            else:
                logger.error("Failed to load ResNet weights")
        else:
            logger.error(f"ResNet weights file not found: {resnet_path}")

    # 加载VRWKV权重
    if use_pretrain:
        model = load_pretrained_ckpt(model)

    return model
    
if __name__ == "__main__":
    model = U_RWKV().cuda()
    model.eval()
    x = torch.randn(1, 3, 224, 224).cuda()
    print(model(x).shape)
