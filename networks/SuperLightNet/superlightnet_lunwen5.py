from __future__ import annotations
from einops import rearrange, repeat

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.utils import InterpolateMode, UpsampleMode
import math
from math import sqrt
from typing import Tuple

class SEfromOther(nn.Module):
    """用另一分支的全局信息做通道缩放"""
    def __init__(self, dim, reduction=2):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x, other):  # x:(B,C,D,H,W), other:(B,C,D,H,W)
        B, C, D, H, W = x.shape
        ctx = other.mean(dim=(2,3,4))               # (B, C)
        s = self.fc2(F.gelu(self.fc1(ctx))).sigmoid().view(B, C, 1, 1, 1)
        return x * s

class BlurPool3D(nn.Module):
    """Anti-aliased downsample: depthwise 3x3x3 blur + stride=2, 无可学习参数"""
    def __init__(self, channels, stride=2):
        super().__init__()
        a = torch.tensor([1., 2., 1.])
        k = (a[:, None, None] * a[None, :, None] * a[None, None, :])
        k = k / k.sum()
        self.register_buffer("kernel", k[None, None, :, :, :])  # (1,1,3,3,3)
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        B, C, _, _, _ = x.shape
        k = self.kernel.expand(C, 1, 3, 3, 3)                  # (C,1,3,3,3)
        return F.conv3d(x, k, stride=self.stride, padding=1, groups=C)

class LightUp3D(nn.Module):
    """Nearest 上采样 + DWConv(3) + PWConv(1) 的轻量上采样"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.dw(x)
        x = F.gelu(x)
        return self.pw(x)

class FusionGate(nn.Module):
    """逐通道自适应融合门：根据(a,b)的全局上下文生成权重"""
    def __init__(self, in_ch_a, in_ch_b, out_ch):
        super().__init__()
        hidden = max(16, out_ch // 2)
        self.fc1 = nn.Linear(in_ch_a + in_ch_b, hidden)
        self.fc2 = nn.Linear(hidden, out_ch)

    def forward(self, a, b):   # a,b: 任意同批量tensor
        ga = a.mean(dim=(2, 3, 4))
        gb = b.mean(dim=(2, 3, 4))
        g = torch.cat([ga, gb], dim=1)
        w = self.fc2(F.gelu(self.fc1(g))).sigmoid().view(g.size(0), -1, 1, 1, 1)
        return w                                                   # (B,out_ch,1,1,1)

class CrossCorrGate(nn.Module):
    """皮尔逊相关通道门：对a,b逐通道计算相关系数后映射为缩放权重"""
    def __init__(self, c_in, hidden=64, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.fc1 = nn.Linear(c_in, hidden)
        self.fc2 = nn.Linear(hidden, c_in)

    def forward(self, a, b):   # a,b: (B,C,D,H,W)
        B, C = a.shape[:2]
        fa = a.view(B, C, -1)
        fb = b.view(B, C, -1)
        ma = fa.mean(-1, keepdim=True)
        mb = fb.mean(-1, keepdim=True)
        sa = fa.std(-1, keepdim=True)
        sb = fb.std(-1, keepdim=True)
        corr = ((fa - ma) * (fb - mb)).mean(-1) / (sa.squeeze(-1) * sb.squeeze(-1) + self.eps)  # (B,C)
        w = self.fc2(F.gelu(self.fc1(corr))).sigmoid().view(B, C, 1, 1, 1)
        return w

class MLP3D_CN(nn.Module):
    """ConvNeXt式3D-MLP: LN(Ch-First) → 1x1 → DW(3) → GELU → 1x1 → LayerScale → DropPath"""
    def __init__(self, dim, expansion=4, layer_scale_init=1e-6, drop_path=0.0):
        super().__init__()
        hidden = dim * expansion
        self.norm = LayerNorm1(dim, data_format="channels_first")
        self.pw1  = nn.Conv3d(dim, hidden, 1)
        self.dw   = nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act  = nn.GELU()
        self.pw2  = nn.Conv3d(hidden, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop  = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        y = self.pw2(self.act(self.dw(self.pw1(self.norm(x)))))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1, 1) * y
        return self.drop(y)

class CrossScaleTokenAttentionMH(nn.Module):
    """
    多头低秩跨尺度注意力：Q来自x2(低分辨率特征图上的每体素)，K/V来自x1↓的稀疏token
    保持O(N*G*C)，但以多头提升表达；对token施加DWConv以增强。
    """
    def __init__(self, c_low, gsize=(2, 2, 2), num_heads=4):
        super().__init__()
        assert c_low % num_heads == 0, "c_low必须能被heads整除"
        self.h = num_heads
        self.d = c_low // num_heads
        self.q = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.k = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.v = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.token_dw = nn.Conv3d(c_low, c_low, 3, padding=1, groups=c_low, bias=False)
        self.proj = nn.Conv3d(c_low, c_low, 1, bias=False)
        self.gsize = gsize
        self.norm_q = LayerNorm1(c_low, data_format="channels_first")
        self.norm_kv = LayerNorm1(c_low, data_format="channels_first")

    def forward(self, x2, x1_low):  # both (B,2C,D2,H2,W2)
        B, C, D2, H2, W2 = x2.shape
        gD, gH, gW = self.gsize
        tokens = F.adaptive_avg_pool3d(x1_low, (gD, gH, gW))
        tokens = tokens + self.token_dw(tokens)                   # 轻量token增强

        q = self.q(self.norm_q(x2)).view(B, self.h, self.d, -1).transpose(2, 3)  # (B,h,N,d)
        k = self.k(self.norm_kv(tokens)).view(B, self.h, self.d, -1)             # (B,h,d,G)
        v = self.v(tokens).view(B, self.h, self.d, -1).transpose(2, 3)           # (B,h,G,d)

        attn = (q @ k) / math.sqrt(self.d)                        # (B,h,N,G)
        attn = attn.softmax(-1)
        out = attn @ v                                            # (B,h,N,d)
        out = out.transpose(2, 3).reshape(B, C, D2, H2, W2)
        return self.proj(out)                                     # (B,2C,D2,H2,W2)

class GatedAxialCrossEnhancePlus(nn.Module):
    """三轴向DWConv混合 + 高分支门控 + SE-from-other"""
    def __init__(self, c_high, k=5):
        super().__init__()
        p = k // 2
        self.ax_d = nn.Conv3d(c_high, c_high, (k, 1, 1), padding=(p, 0, 0), groups=c_high, bias=False)
        self.ax_h = nn.Conv3d(c_high, c_high, (1, k, 1), padding=(0, p, 0), groups=c_high, bias=False)
        self.ax_w = nn.Conv3d(c_high, c_high, (1, 1, k), padding=(0, 0, p), groups=c_high, bias=False)
        self.mix  = nn.Conv3d(c_high, c_high, 1, bias=False)

        self.gate_gen = nn.Sequential(
            nn.Conv3d(c_high, c_high, 3, padding=1, groups=c_high, bias=False),
            nn.Conv3d(c_high, c_high, 1, bias=False),
            nn.Sigmoid()
        )
        self.se_from_other = SEfromOther(c_high, reduction=4)

    def forward(self, x1, x2_up):
        y = self.mix(self.ax_d(x1) + self.ax_h(x1) + self.ax_w(x1))
        g = self.gate_gen(x2_up)
        y = y * (1.0 + g)
        y = self.se_from_other(y, x2_up)
        return y

# ---------- 新的 CSDI3D（drop-in） ----------

class CSDI3D(nn.Module):
    def __init__(self, C, token_grid=(2, 2, 2), mlp_expansion=4, heads=4, drop_path=0.0):
        super().__init__()
        # 下采样：抗混叠 + 1x1 投影到2C
        self.down_x1 = nn.Sequential(
            BlurPool3D(C, stride=2),
            nn.Conv3d(C, 2 * C, kernel_size=1, bias=False)
        )
        # 上采样：轻量上采样至C
        self.proj_x2_up = LightUp3D(2 * C, C)

        # 分支1：多头低秩跨尺度注意力（低分侧）
        self.csta = CrossScaleTokenAttentionMH(c_low=2 * C, gsize=token_grid, num_heads=heads)

        # 分支2：改进的高分侧轴向增强
        self.gace = GatedAxialCrossEnhancePlus(c_high=C)

        # 分支3：皮尔逊相关统计门
        self.cc_low  = CrossCorrGate(c_in=2 * C, hidden=2 * C)
        self.cc_high = CrossCorrGate(c_in=C,     hidden=2 * C)

        # 融合与投影
        self.fuse_low_skip         = nn.Conv3d(2 * C, 2 * C, kernel_size=1, bias=False)
        self.fuse_high_from_lowup  = nn.Conv3d(2 * C, C,     kernel_size=1, bias=False)

        # 动态融合门
        self.gate_low  = FusionGate(in_ch_a=2 * C, in_ch_b=2 * C, out_ch=2 * C)
        self.gate_high = FusionGate(in_ch_a=C,     in_ch_b=2 * C, out_ch=C)

        # MLP 头（保持“仅与MLP(x)相加”的残差形式）
        self.mlp1 = MLP3D_CN(C,     expansion=mlp_expansion, drop_path=drop_path)
        self.mlp2 = MLP3D_CN(2 * C, expansion=mlp_expansion, drop_path=drop_path)

    def forward(self, x1, x2):
        """
        x1: (B, C,  D,   H,   W)
        x2: (B,2C, D/2, H/2, W/2)
        """
        B, C, D, H, W = x1.shape
        _, C2, D2, H2, W2 = x2.shape
        assert C2 == 2 * C and D2 * 2 == D and H2 * 2 == H and W2 * 2 == W, "Shape mismatch."

        # 准备跨尺度特征
        x1_low = self.down_x1(x1)                         # (B,2C,D/2,H/2,W/2)
        x2_up  = self.proj_x2_up(x2)                      # (B,C, D,  H,  W)

        # 分支1：低分侧跨尺度注意力（Q:x2, K/V:x1_low token）
        y2_A = self.csta(x2, x1_low)                      # (B,2C,D/2,H/2,W/2)

        # 分支2：高分侧轴向增强（由x2_up门控+SE）
        y1_B = self.gace(x1, x2_up)                       # (B,C,D,H,W)

        # 分支3：跨路统计一致性校准（皮尔逊相关）
        w_low  = self.cc_low (a=x1_low, b=x2)             # (B,2C,1,1,1)
        w_high = self.cc_high(a=x1,     b=x2_up)          # (B,C,1,1,1)
        y2_A = y2_A * w_low
        y1_B = y1_B * w_high

        # 低分侧融合：注意力 vs 旁路 的自适应加权
        gate_low = self.gate_low(x1_low, x2)              # (B,2C,1,1,1)
        y2 = gate_low * y2_A + (1.0 - gate_low) * self.fuse_low_skip(x1_low)

        # 高分侧融合：来自低分的上采样信息 vs 高分增强
        y2_up_feat = F.interpolate(y2, scale_factor=2, mode='trilinear', align_corners=False)  # (B,2C,D,H,W)
        y1_from_low = self.fuse_high_from_lowup(y2_up_feat)                                     # (B,C,D,H,W)
        gate_high = self.gate_high(x1, y2)                                                     # (B,C,1,1,1)
        y1 = gate_high * y1_B + (1.0 - gate_high) * y1_from_low

        # MLP 残差头（与原设计一致：y += MLP(x)）
        y1_final = y1 + self.mlp1(x1)
        y2_final = y2 + self.mlp2(x2)
        return y1_final, y2_final

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, input_x):
        if self.data_format == "channels_last":
            return F.layer_norm(input_x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = input_x.mean(1, keepdim=True)
            s = (input_x - u).pow(2).mean(1, keepdim=True)
            input_x = (input_x - u) / torch.sqrt(s + self.eps)
            input_x = self.weight[:, None, None] * input_x + self.bias[:, None, None]
            return input_x
        return None
    
class LayerNorm1(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    # by-EGE-UNET
    def __init__(self, dim_in, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_in, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x4 = self.dw(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class THPAEncFR3(nn.Module):
    def __init__(self, in_channels, expr):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels // 2)
        self.GHPA_dim = Grouped_multi_axis_Hadamard_Product_Attention(in_channels // 2, in_channels // 2)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.mlp = MlpChannel(in_channels, expr)

    def forward(self, input_x: Tensor, dummy_tensor=None):
        input_x, x_residual = torch.chunk(input_x, 2, dim=1)
        input_x = self.norm1(input_x)
        B, C, W, H, D = input_x.shape

        random_direction = torch.randint(0, 3, (1,)).item()
        if random_direction == 0:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (h b) c w d")),
                                "(h b) c w d -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (h b) c w d").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(h b) c w d -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        elif random_direction == 1:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (w b) c h d")),
                                "(w b) c h d -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (w b) c h d").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(w b) c h d -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        elif random_direction == 2:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (d b) c w h")),
                                "(d b) c w h -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (d b) c w h").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(d b) c w h -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        else:
            raise NotImplementedError
        input_x = torch.cat((WHD_dim, x_residual), dim=1)
        input_x = self.norm2(input_x)
        input_x = self.mlp(input_x)
        return input_x

class NormDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self,input, dummy_tensor=None):
        return self.proj(self.norm(input))

class Learnable_Res_Skip_UpRepr4(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims = 3):
        super().__init__()
        self.upc = Convolution(
            spatial_dims=spatial_dims, in_channels = in_channels, out_channels = out_channels, strides=1,
            kernel_size=1, bias=False, conv_only=True
        )
        self.ups = UpSample(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            scale_factor=2,
            mode=UpsampleMode.NONTRAINABLE,
            interp_mode=InterpolateMode.LINEAR,
            align_corners=False,
        )

        self.repr_mldw = nn.Sequential(Convolution(spatial_dims=spatial_dims, in_channels=out_channels,
                                                   out_channels=out_channels, strides=1,
                                                   kernel_size=3, bias=False, conv_only=True, groups=out_channels // 12),
                                       nn.GELU(),
                                       Convolution(spatial_dims=spatial_dims, in_channels=out_channels,
                                                   out_channels=out_channels,
                                                   strides=1, kernel_size=1, bias=False, conv_only=True, groups=1)
                                       )

        self.norm = nn.InstanceNorm3d(out_channels)
        self.group_skip_scale = nn.Parameter(torch.Tensor(1, out_channels, 1, 1, 1), requires_grad=True)
        nn.init.ones_(self.group_skip_scale)
        self.group_res_scale = nn.Parameter(torch.Tensor(1), requires_grad=True)
        nn.init.ones_(self.group_res_scale)

    def forward(self, inp_skip, dummy_tensor=None):
        input, skip = inp_skip
        input = self.ups(self.upc(input))
        input = input + skip * self.group_skip_scale
        res = input

        input = self.norm(input)
        out = self.repr_mldw(input)

        return out + res * self.group_res_scale

class MlpChannel(nn.Module):
    def __init__(self, in_channels, expr = 1, out_channels = None):
        if out_channels is None:
            out_channels = in_channels
        super().__init__()
        self.fc1 = nn.Conv3d(in_channels, in_channels * expr, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(in_channels * expr, out_channels, 1)

    def forward(self, input_x):
        input_x = self.fc1(input_x)
        input_x = self.act(input_x)
        input_x = self.fc2(input_x)
        return input_x

def block_creator(coder_str, depths_unidirectional, in_channels, out_channels=0):
    if out_channels == 0:
        out_channels = in_channels

    if coder_str == "NormDownsample":
        block = NormDownsample(in_channels, out_channels)
    elif coder_str == "THPAEncFR3":
        block = nn.Sequential(*[
            THPAEncFR3(in_channels,expr=2)
            for _ in range(depths_unidirectional)
        ])
    elif coder_str == "Learnable_Res_Skip_UpRepr4":
        block = Learnable_Res_Skip_UpRepr4(in_channels,out_channels)
    else:
        print("encoder error")
        raise NotImplementedError
    return block

class JCMNetv8Enc(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels=32,
                 class_nums=4,
                 checkpoint_style="",
                 expr=2,
                 depths_unidirectional=None,
                 ):
        super(JCMNetv8Enc, self).__init__()

        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False

        if depths_unidirectional is None:
            raise NotImplementedError
        elif depths_unidirectional == "small":
            depths_unidirectional = [1, 1, 2, 2, 2]
        elif depths_unidirectional == "medium":
            depths_unidirectional = [3, 4, 4, 4, 4]
        elif depths_unidirectional == "large":
            depths_unidirectional = [3, 4, 8, 8, 8]

        encoder = ["THPAEncFR3", "THPAEncFR3", "THPAEncFR3", "THPAEncFR3", "THPAEncFR3"]


        downcoder = "NormDownsample"

        self.stem = nn.Conv3d(init_channels, n_channels, kernel_size=1)

        self.repr_block_0 = block_creator(encoder[0], depths_unidirectional[0], n_channels)
        self.dwn_block_0 = block_creator(downcoder, 1, n_channels, n_channels * 2)

        self.repr_block_1 = block_creator(encoder[1], depths_unidirectional[1], n_channels * 2)
        self.dwn_block_1 = block_creator(downcoder, 1, n_channels * 2, n_channels * 4)

        self.repr_block_2 = block_creator(encoder[2], depths_unidirectional[2], n_channels * 4)
        self.dwn_block_2 = block_creator(downcoder, 1, n_channels * 4, n_channels * 8)

        self.repr_block_3 = block_creator(encoder[3], depths_unidirectional[3], n_channels * 8)
        self.dwn_block_3 = block_creator(downcoder, 1, n_channels * 8, n_channels * 16)

        self.emb_block = block_creator(encoder[4], depths_unidirectional[4], n_channels * 16)

        self.csdi_1 = CSDI3D(C=n_channels, token_grid=(2,2,2), mlp_expansion=4, heads=2)
        
        self.csdi_2 = CSDI3D(C=n_channels*4, token_grid=(2,2,2), mlp_expansion=4, heads=4)


        if self.outside_block_checkpointing:
        # Used to fix PyTorch checkpointing bug from MedNeXt
            self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    # by-MedNeXt
    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor, use_reentrant=True)
        return x

    def forward(self, input: Tensor):
        if self.outside_block_checkpointing:

            pass
        else:
            input = self.stem(input)
            skips = []
            repr0 = self.repr_block_0(input)
            dwn0 = self.dwn_block_0(repr0)
            # skips.append(repr0)
            # del repr0

            repr1 = self.repr_block_1(dwn0)
            dwn1 = self.dwn_block_1(repr1)
            # skips.append(repr1)
            # del repr1

            repr2 = self.repr_block_2(dwn1)
            dwn2 = self.dwn_block_2(repr2)
            # skips.append(repr2)
            # del repr2

            repr3 = self.repr_block_3(dwn2)
            dwn3 = self.dwn_block_3(repr3)
            
            repr0,repr1 = self.csdi_1(repr0,repr1)
            repr2,repr3 = self.csdi_2(repr2,repr3)


            skips.append(repr0)
            skips.append(repr1)
            skips.append(repr2)
            skips.append(repr3)
            # del repr3

            hidden = self.emb_block(dwn3)

            return hidden, tuple(skips)

class JCMNetv8Dec(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels=32,
                 class_nums=4,
                 checkpoint_style="",
                 expr=2,
                 depths_unidirectional=None,
                 ):
        super(JCMNetv8Dec, self).__init__()

        if depths_unidirectional is None:
            raise NotImplementedError
        elif depths_unidirectional == "small":
            depths_unidirectional = [1, 1, 2, 2, 2]
        elif depths_unidirectional == "medium":
            depths_unidirectional = [3, 4, 4, 4, 4]
        elif depths_unidirectional == "large":
            depths_unidirectional = [3, 4, 8, 8, 8]

        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False

        decoder = ["Learnable_Res_Skip_UpRepr4","Learnable_Res_Skip_UpRepr4",
                   "Learnable_Res_Skip_UpRepr4","Learnable_Res_Skip_UpRepr4"]

        self.repr_block_up_3 = block_creator(decoder[3],depths_unidirectional[3],n_channels * 16,n_channels * 8)
        self.repr_block_up_2 = block_creator(decoder[2],depths_unidirectional[2],n_channels * 8,n_channels * 4)
        self.repr_block_up_1 = block_creator(decoder[1],depths_unidirectional[1],n_channels * 4,n_channels * 2)
        self.repr_block_up_0 = block_creator(decoder[0],depths_unidirectional[0],n_channels * 2,n_channels)

        if self.outside_block_checkpointing:
        # Used to fix PyTorch checkpointing bug from MedNeXt
            self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    # by-MedNeXt
    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor, use_reentrant=True)
        return x

    def forward(self, hidden, skips):
        if self.outside_block_checkpointing:
            pass

        else:
            dec = self.repr_block_up_3((hidden,skips[3]))
            dec = self.repr_block_up_2((dec,skips[2]))
            dec = self.repr_block_up_1((dec,skips[1]))
            dec = self.repr_block_up_0((dec,skips[0]))

            return dec

class NormalU_Net_lunwen5(nn.Module):
    def __init__(self,
                 init_channels = 4,
                 n_channels = 24,
                 class_nums = 4,
                 checkpoint_style = "",
                 expr = 2,
                 depths_unidirectional=None,
                 ):
        super().__init__()
        args_list = [init_channels,
                     n_channels,
                     class_nums,
                     checkpoint_style,
                     expr,
                     depths_unidirectional]
        self.ParallelU_Net_enc_m = JCMNetv8Enc(*args_list)
        self.ParallelU_Net_dec_m = JCMNetv8Dec(*args_list)

        self.norm = nn.GroupNorm(n_channels, n_channels)
        self.proj = MlpChannel(n_channels, expr, class_nums)

    def forward(self, input, dummy_tensor=None):
        hidden_m, skips_m = self.ParallelU_Net_enc_m(input)
        out = self.ParallelU_Net_dec_m(hidden_m, skips_m)
        out = self.proj(self.norm(out))
        return out


if __name__ == '__main__':
    cuda0 = torch.device('cuda:0')
    x = torch.rand((1, 1, 96, 96, 96), device=cuda0)
    model = NormalU_Net(depths_unidirectional='small')
    model.cuda()
    print(model)
    print(str(sum([param.nelement() for param in model.parameters()]) / 1e6) + 'M')