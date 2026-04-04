import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union

try:
    from timm.models.layers import trunc_normal_, DropPath  # noqa: F401
except Exception:
    trunc_normal_, DropPath = None, None  # type: ignore

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

import math


def _eca_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    """ECA paper heuristic for 1D conv kernel size."""
    if channels <= 0:
        return 3
    t = int(abs((math.log2(channels) / gamma) + b))
    k = t if t % 2 else t + 1
    return max(k, 3)


class ECAGate(nn.Module):
    """
    Efficient Channel Attention (ECA) gate.
    Return per-channel weights (do not multiply unless you do outside).
    Support 2D/3D.
    """
    def __init__(self, channels: int, dims: int = 3, k_size: Optional[int] = None):
        super().__init__()
        assert dims in (2, 3)
        self.dims = dims
        k = _eca_kernel_size(channels) if k_size is None else k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,...) -> w: (B,C,1,1[,1])
        if self.dims == 2:
            y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B,C)
        else:
            y = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (B,C)
        y = self.conv(y.unsqueeze(1)).squeeze(1)  # (B,C)
        w = torch.sigmoid(y)
        if self.dims == 2:
            return w[:, :, None, None]
        return w[:, :, None, None, None]


class ECABlock(nn.Module):
    """ECA that directly reweights input."""
    def __init__(self, channels: int, dims: int = 3, k_size: Optional[int] = None):
        super().__init__()
        self.gate = ECAGate(channels, dims=dims, k_size=k_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class AntiAliasedDownsample2D(nn.Module):
    """
    Parameter-free blur + pool downsampling.
    Keep pooling floor behavior for odd sizes (same as avg/max pool2d).
    """
    def __init__(self, channels: int, filt_size: int = 3, pool: str = "avg"):
        super().__init__()
        assert filt_size in (3, 5), "only support 3 or 5"
        assert pool in ("avg", "max")
        if filt_size == 3:
            a = torch.tensor([1., 2., 1.])
        else:
            a = torch.tensor([1., 4., 6., 4., 1.])
        filt = (a[:, None] * a[None, :])
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))  # (C,1,k,k)
        self.channels = channels
        self.pad = (filt_size - 1) // 2
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.filt, stride=1, padding=self.pad, groups=self.channels)
        if self.pool == "max":
            return F.max_pool2d(x, kernel_size=2, stride=2)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class SepConv2d(nn.Module):
    """Depthwise-separable Conv2d + IN + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, bias: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class AxisSepConv3d(nn.Module):
    """Axis-aligned depthwise-separable Conv3d + IN + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: tuple, stride: tuple, padding: tuple, bias: bool = False):
        super().__init__()
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias)
        self.pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        else:
            raise NotImplementedError(f"Unsupported data_format: {self.data_format}")


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat: int = 96,  # kept for compatibility
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)
        if x_e is not None:
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))
        else:
            x = self.convs(x_0)
        return x


def haar_2d(x: torch.Tensor):
    """
    x: (N, C, H, W)
    return: LL, LH, HL, HH: (N, C, H/2, W/2)
    """
    N, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, 

    H_even = x[:, :, 0::2, :]
    H_odd = x[:, :, 1::2, :]
    L = (H_even + H_odd) / 2.0
    Hc = (H_even - H_odd) / 2.0

    W_even_L = L[:, :, :, 0::2]
    W_odd_L = L[:, :, :, 1::2]
    LL = (W_even_L + W_odd_L) / 2.0
    LH = (W_even_L - W_odd_L) / 2.0

    W_even_H = Hc[:, :, :, 0::2]
    W_odd_H = Hc[:, :, :, 1::2]
    HL = (W_even_H + W_odd_H) / 2.0
    HH = (W_even_H - W_odd_H) / 2.0
    return LL, LH, HL, HH


def ihaar_2d(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
    """
    Inverse Haar:
    LL,LH,HL,HH: (N,C,h,w) -> (N,C,2h,2w)
    """
    N, C, h, w = LL.shape
    device, dtype = LL.device, LL.dtype

    W_even_L = LL + LH
    W_odd_L = LL - LH
    W_even_H = HL + HH
    W_odd_H = HL - HH

    L = torch.zeros((N, C, h, 2 * w), device=device, dtype=dtype)
    Hc = torch.zeros((N, C, h, 2 * w), device=device, dtype=dtype)
    L[:, :, :, 0::2] = W_even_L
    L[:, :, :, 1::2] = W_odd_L
    Hc[:, :, :, 0::2] = W_even_H
    Hc[:, :, :, 1::2] = W_odd_H

    x = torch.zeros((N, C, 2 * h, 2 * w), device=device, dtype=dtype)
    x[:, :, 0::2, :] = L + Hc
    x[:, :, 1::2, :] = L - Hc
    return x


def coords_2d(h: int, w: int, device, dtype):
    yy = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(1, 1, h, w)
    xx = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(1, 1, h, w)
    return torch.cat([yy, xx], dim=1)  # (1,2,h,w)



class WaveletGatingLite(nn.Module):
    """
    Ultra-light 4-subband gating:
      - per-channel energy (abs-mean)
      - shared learnable 4x4 affine (20 params)
      - softmax across bands
    """
    def __init__(self, init_identity: bool = True):
        super().__init__()
        self.affine = nn.Linear(4, 4, bias=True)
        if init_identity:
            nn.init.eye_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)

    def forward(self, LL, LH, HL, HH):
        N, C, _, _ = LL.shape
        v = torch.stack(
            [
                LL.abs().mean(dim=(2, 3)),
                LH.abs().mean(dim=(2, 3)),
                HL.abs().mean(dim=(2, 3)),
                HH.abs().mean(dim=(2, 3)),
            ],
            dim=-1,
        )  # (N,C,4)
        w = self.affine(v.reshape(-1, 4))
        w = torch.softmax(w, dim=-1).view(N, C, 4)
        w = w.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (N,4,C,1,1)
        return LL * w[:, 0], LH * w[:, 1], HL * w[:, 2], HH * w[:, 3]



class ViewGradAttentionDown(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max((3 * channels + 3) // reduction, 16)
        self.fc1 = nn.Linear(3 * channels + 3, hidden)
        self.fc2 = nn.Linear(hidden, 3 * channels)

    @staticmethod
    def _axis_grad_prior(x: torch.Tensor, dim: int) -> torch.Tensor:
        g = torch.abs(x.diff(dim=dim)).mean(dim=(1, 2, 3, 4), keepdim=False)  # (B,)
        return g.unsqueeze(1)  # (B,1)

    def forward(self, y_d, y_h, y_w, x_in):
        B, C, _, _, _ = y_d.shape
        s_d = F.adaptive_avg_pool3d(y_d, 1).view(B, C)
        s_h = F.adaptive_avg_pool3d(y_h, 1).view(B, C)
        s_w = F.adaptive_avg_pool3d(y_w, 1).view(B, C)

        gD = self._axis_grad_prior(x_in, dim=2)
        gH = self._axis_grad_prior(x_in, dim=3)
        gW = self._axis_grad_prior(x_in, dim=4)

        feat = torch.cat([s_d, s_h, s_w, gD, gH, gW], dim=1)
        a = self.fc2(F.relu(self.fc1(feat)))
        a = a.view(B, 3, C, 1, 1, 1)
        a = torch.softmax(a, dim=1)
        y = a[:, 0] * y_d + a[:, 1] * y_h + a[:, 2] * y_w
        return y, a



class TriPlaneFreqAttDown(nn.Module):
    """
    Input:  x (B,C,D,H,W)
    Output: y (B,out_channels,D/2,H/2,W/2)
    """
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, reduction: int = 4, view_drop_p: float = 0.0):
        super().__init__()
        if out_channels is None:
            out_channels = 2 * in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels
        self.view_drop_p = view_drop_p

        mid = self.mid_channels

        # D-view: (H,W) plane per depth slice
        self.pre2d_d = SepConv2d(in_channels + 2, mid)
        self.down2d_d = AntiAliasedDownsample2D(mid, filt_size=3, pool="avg")
        self.conv3d_d = AxisSepConv3d(mid, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        # H-view: (D,W) plane per height slice
        self.pre2d_h = SepConv2d(in_channels + 2, mid)
        self.down2d_h = AntiAliasedDownsample2D(mid, filt_size=3, pool="avg")
        self.conv3d_h = AxisSepConv3d(mid, out_channels, kernel_size=(1, 3, 1), stride=(1, 2, 1), padding=(0, 1, 0))

        # W-view: (D,H) plane per width slice + wavelet
        self.pre2d_w = SepConv2d(in_channels + 2, mid)
        self.wavelet_gate = WaveletGatingLite(init_identity=True)
        self.fuse_w = nn.Sequential(
            nn.Conv2d(mid * 4, mid, kernel_size=1, bias=False),
            nn.InstanceNorm2d(mid, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv3d_w = AxisSepConv3d(mid, out_channels, kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))

        # view attention with axis prior
        self.view_attn = ViewGradAttentionDown(out_channels, reduction=reduction)

        # axis refinement (DW + PW)
        self.axis_dw_d = nn.Conv3d(out_channels, out_channels, (3, 1, 1), padding=(1, 0, 0), groups=out_channels, bias=False)
        self.axis_dw_h = nn.Conv3d(out_channels, out_channels, (1, 3, 1), padding=(0, 1, 0), groups=out_channels, bias=False)
        self.axis_dw_w = nn.Conv3d(out_channels, out_channels, (1, 1, 3), padding=(0, 0, 1), groups=out_channels, bias=False)
        self.axis_pw = nn.Conv3d(out_channels, out_channels, 1, bias=False)
        self.axis_bn = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.eca = ECABlock(out_channels, dims=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        device, dtype = x.device, x.dtype
        mid = self.mid_channels

        # D-view
        x_d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        c_hw = coords_2d(H, W, device, dtype).expand(B * D, -1, -1, -1)
        x_d = self.pre2d_d(torch.cat([x_d, c_hw], dim=1))
        x_d = self.down2d_d(x_d)  # (B*D,mid,H/2,W/2)
        x_d = x_d.view(B, D, mid, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        x_d = self.conv3d_d(x_d)  # (B,out,D/2,H/2,W/2)

        # H-view
        x_h = x.permute(0, 3, 1, 2, 4).reshape(B * H, C, D, W)
        c_dw = coords_2d(D, W, device, dtype).expand(B * H, -1, -1, -1)
        x_h = self.pre2d_h(torch.cat([x_h, c_dw], dim=1))
        x_h = self.down2d_h(x_h)  # (B*H,mid,D/2,W/2)
        x_h = x_h.view(B, H, mid, D // 2, W // 2).permute(0, 2, 3, 1, 4)
        x_h = self.conv3d_h(x_h)  # (B,out,D/2,H/2,W/2)

        # W-view + wavelet
        x_w = x.permute(0, 4, 1, 2, 3).reshape(B * W, C, D, H)
        c_dh = coords_2d(D, H, device, dtype).expand(B * W, -1, -1, -1)
        x_w = self.pre2d_w(torch.cat([x_w, c_dh], dim=1))  # (B*W,mid,D,H)

        LL, LH, HL, HH = haar_2d(x_w)
        LL, LH, HL, HH = self.wavelet_gate(LL, LH, HL, HH)
        x_w = self.fuse_w(torch.cat([LL, LH, HL, HH], dim=1))
        x_w = x_w.view(B, W, mid, D // 2, H // 2).permute(0, 2, 3, 4, 1)
        x_w = self.conv3d_w(x_w)  # (B,out,D/2,H/2,W/2)

        # ViewDrop (optional)
        if self.training and self.view_drop_p > 0.0 and torch.rand(1, device=device) < self.view_drop_p:
            drop_id = torch.randint(0, 3, (1,), device=device).item()
            if drop_id == 0:
                x_d = torch.zeros_like(x_d)
            elif drop_id == 1:
                x_h = torch.zeros_like(x_h)
            else:
                x_w = torch.zeros_like(x_w)

        # view attention fusion
        x_va, _ = self.view_attn(x_d, x_h, x_w, x)

        # axis refine
        axis = self.axis_dw_d(x_va) + self.axis_dw_h(x_va) + self.axis_dw_w(x_va)
        axis = self.axis_pw(axis)
        axis = self.axis_bn(axis)
        axis = self.act(axis)
        out = x_va + axis

        out = self.eca(out)
        return out


class BandGate3Lite(nn.Module):
    """Ultra-light 3-band gating (12 params total)."""
    def __init__(self, init_identity: bool = True):
        super().__init__()
        self.affine = nn.Linear(3, 3, bias=True)
        if init_identity:
            nn.init.eye_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)

    def forward(self, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
        N, C, _, _ = LH.shape
        v = torch.stack(
            [
                LH.abs().mean(dim=(2, 3)),
                HL.abs().mean(dim=(2, 3)),
                HH.abs().mean(dim=(2, 3)),
            ],
            dim=-1,
        )  # (N,C,3)
        w = self.affine(v.reshape(-1, 3))
        w = torch.softmax(w, dim=-1).view(N, C, 3)
        w = w.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (N,3,C,1,1)
        return LH * w[:, 0], HL * w[:, 1], HH * w[:, 2]


class PlaneWaveletLifting2x(nn.Module):
    
    def __init__(self, in_ch: int, out_ch: int, reduction: int = 4):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch + 2, in_ch + 2, 3, padding=1, groups=in_ch + 2, bias=False),
            nn.Conv2d(in_ch + 2, in_ch, 1, bias=False),
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.eca = ECABlock(in_ch, dims=2)

        self.to_ll = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        self.hf_dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.hf_norm = nn.InstanceNorm2d(in_ch, affine=True)
        self.hf_pw = nn.Conv2d(in_ch, out_ch * 3, 1, bias=False)

        self.band_gate = BandGate3Lite(init_identity=True)

        self.post = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    @staticmethod
    def _edge_confidence_map(LL: torch.Tensor) -> torch.Tensor:
        # LL: (N,C,h,w) -> conf: (N,1,h,w) in [0,1]
        dx = F.pad(LL[:, :, :, 1:] - LL[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(LL[:, :, 1:, :] - LL[:, :, :-1, :], (0, 0, 0, 1))
        g = (dx.abs() + dy.abs()).mean(dim=1, keepdim=True)  # (N,1,h,w)
        mean = g.mean(dim=(2, 3), keepdim=True)
        std = g.std(dim=(2, 3), keepdim=True) + 1e-6
        conf = torch.sigmoid((g - mean) / std)
        return conf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, Cin, h, w = x.shape
        c = coords_2d(h, w, x.device, x.dtype).expand(N, -1, -1, -1)

        x = self.pre(torch.cat([x, c], dim=1))
        x = self.eca(x)

        LL = self.to_ll(x)

        hf = self.hf_dw(x)
        hf = self.hf_norm(hf)
        hf = F.leaky_relu(hf, negative_slope=0.1, inplace=True)
        HF = self.hf_pw(hf)

        LH, HL, HH = torch.chunk(HF, 3, dim=1)
        LH, HL, HH = self.band_gate(LH, HL, HH)

        y_wave = ihaar_2d(LL, LH, HL, HH)  # (N,Cout,2h,2w)
        y_interp = F.interpolate(LL, scale_factor=2, mode="bilinear", align_corners=False)

        conf = self._edge_confidence_map(LL)
        conf_up = F.interpolate(conf, scale_factor=2, mode="bilinear", align_corners=False)
        y = conf_up * y_wave + (1.0 - conf_up) * y_interp

        y = self.post(y)
        return y


class AxisUp1D(nn.Module):
    def __init__(self, ch: int, axis: str):
        super().__init__()
        if axis == "D":
            k, s = (2, 1, 1), (2, 1, 1)
        elif axis == "H":
            k, s = (1, 2, 1), (1, 2, 1)
        elif axis == "W":
            k, s = (1, 1, 2), (1, 1, 2)
        else:
            raise ValueError("axis must be one of {'D','H','W'}")

        self.dw = nn.ConvTranspose3d(ch, ch, kernel_size=k, stride=s, padding=0, groups=ch, bias=False)
        self.norm = nn.InstanceNorm3d(ch, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pw = nn.Conv3d(ch, ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pw(x)
        return x


class TriViewGradAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max((3 * ch + 3) // reduction, 16)
        self.fc1 = nn.Linear(3 * ch + 3, hidden)
        self.fc2 = nn.Linear(hidden, 3 * ch)

    @staticmethod
    def _axis_grad_prior(x: torch.Tensor, dim: int) -> torch.Tensor:
        d = torch.abs(x.diff(dim=dim)).mean(dim=(1, 2, 3, 4), keepdim=False)
        return d.unsqueeze(1)

    def forward(self, y_d, y_h, y_w, x_low):
        B, C, _, _, _ = y_d.shape
        s_d = F.adaptive_avg_pool3d(y_d, 1).view(B, C)
        s_h = F.adaptive_avg_pool3d(y_h, 1).view(B, C)
        s_w = F.adaptive_avg_pool3d(y_w, 1).view(B, C)

        gD = self._axis_grad_prior(x_low, dim=2)
        gH = self._axis_grad_prior(x_low, dim=3)
        gW = self._axis_grad_prior(x_low, dim=4)

        feat = torch.cat([s_d, s_h, s_w, gD, gH, gW], dim=1)
        a = self.fc2(F.relu(self.fc1(feat)))
        a = a.view(B, 3, C, 1, 1, 1)
        a = torch.softmax(a, dim=1)
        y = a[:, 0] * y_d + a[:, 1] * y_h + a[:, 2] * y_w
        return y, a


class AnisoTriPlaneWaveletUp(nn.Module):
    """
    Input:  (B, 2*C, D/2, H/2, W/2)
    Output: (B,   C, D,   H,   W)    -> tri residual-like
    """
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, reduction: int = 4, view_drop_p: float = 0.0):
        super().__init__()
        out_channels = out_channels or (in_channels // 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.view_drop_p = view_drop_p

        self.plane_d = PlaneWaveletLifting2x(in_channels, out_channels, reduction=reduction)  # up (H,W)
        self.plane_h = PlaneWaveletLifting2x(in_channels, out_channels, reduction=reduction)  # up (D,W)
        self.plane_w = PlaneWaveletLifting2x(in_channels, out_channels, reduction=reduction)  # up (D,H)

        self.up_d = AxisUp1D(out_channels, axis="D")
        self.up_h = AxisUp1D(out_channels, axis="H")
        self.up_w = AxisUp1D(out_channels, axis="W")

        self.view_attn = TriViewGradAttention(out_channels, reduction=reduction)

        self.axis_dw_d = nn.Conv3d(out_channels, out_channels, (3, 1, 1), padding=(1, 0, 0), groups=out_channels, bias=False)
        self.axis_dw_h = nn.Conv3d(out_channels, out_channels, (1, 3, 1), padding=(0, 1, 0), groups=out_channels, bias=False)
        self.axis_dw_w = nn.Conv3d(out_channels, out_channels, (1, 1, 3), padding=(0, 0, 1), groups=out_channels, bias=False)
        self.axis_pw = nn.Conv3d(out_channels, out_channels, 1, bias=False)
        self.axis_norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.eca = ECABlock(out_channels, dims=3)

        self.res_scale_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)=0.119

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, D2, H2, W2 = x.shape
        assert Cin == self.in_channels, f"expected Cin={self.in_channels}, got {Cin}"

        # D-branch
        xd = x.permute(0, 2, 1, 3, 4).reshape(B * D2, Cin, H2, W2)
        yd2 = self.plane_d(xd)
        yd3 = yd2.view(B, D2, self.out_channels, 2 * H2, 2 * W2).permute(0, 2, 1, 3, 4)
        y_d = self.up_d(yd3)

        # H-branch
        xh = x.permute(0, 3, 1, 2, 4).reshape(B * H2, Cin, D2, W2)
        yh2 = self.plane_h(xh)
        yh3 = yh2.view(B, H2, self.out_channels, 2 * D2, 2 * W2).permute(0, 2, 3, 1, 4)
        y_h = self.up_h(yh3)

        # W-branch
        xw = x.permute(0, 4, 1, 2, 3).reshape(B * W2, Cin, D2, H2)
        yw2 = self.plane_w(xw)
        yw3 = yw2.view(B, W2, self.out_channels, 2 * D2, 2 * H2).permute(0, 2, 3, 4, 1)
        y_w = self.up_w(yw3)

        # ViewDrop
        if self.training and self.view_drop_p > 0.0 and torch.rand(1, device=x.device) < self.view_drop_p:
            drop_id = torch.randint(0, 3, (1,), device=x.device).item()
            if drop_id == 0:
                y_d = torch.zeros_like(y_d)
            elif drop_id == 1:
                y_h = torch.zeros_like(y_h)
            else:
                y_w = torch.zeros_like(y_w)

        # fuse
        y_tri, _ = self.view_attn(y_d, y_h, y_w, x)

        # axis refine
        axis = self.axis_dw_d(y_tri) + self.axis_dw_h(y_tri) + self.axis_dw_w(y_tri)
        axis = self.axis_pw(axis)
        axis = self.axis_norm(axis)
        axis = self.act(axis)
        y_tri = y_tri + axis

        y_tri = self.eca(y_tri)

        scale = torch.sigmoid(self.res_scale_logit)
        return scale * y_tri



class TSPD(nn.Module):
    """
    Trihedral Spectro‑Projective Decimation (TSPD)
    """
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        tri_reduction: int = 4,
        tri_view_drop_p: float = 0.0,
    ):
        super().__init__()
        self.down = Down(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.tri = TriPlaneFreqAttDown(in_channels=in_chns, out_channels=out_chns, reduction=tri_reduction, view_drop_p=tri_view_drop_p)

        self.tri_gate_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid=0.5
        self.eca = ECABlock(out_chns, dims=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_base = self.down(x)
        x_tri = self.tri(x)
        gate = torch.sigmoid(self.tri_gate_logit)
        out = x_base + gate * x_tri
        out = self.eca(out)
        return out


class TSPP(nn.Module):
    """
    Trihedral Spectro‑Projective Prolongation (TSPP)
    """
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        tri_reduction: int = 4,
        tri_view_drop_p: float = 0.0,
        halves: bool = True,
    ):
        super().__init__()
        self.upcat = UpCat(
            spatial_dims=spatial_dims,
            in_chns=in_chns,
            cat_chns=cat_chns,
            out_chns=out_chns,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
            halves=halves,
        )
        self.tri_up = AnisoTriPlaneWaveletUp(in_channels=in_chns, out_channels=out_chns, reduction=tri_reduction, view_drop_p=tri_view_drop_p)

        self.base_gate = ECAGate(out_chns, dims=3)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]) -> torch.Tensor:
        y_base = self.upcat(x, x_e)
        y_tri = self.tri_up(x)  
        g = self.base_gate(y_base)
        out = y_base + y_tri * g
        return out



class SVRUNet(nn.Module):

    "SVR‑UNet（Spectro‑View Routed U‑Net）"

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (24, 48, 96, 192, 384, 24),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
        tri_reduction: int = 4,
        tri_view_drop_p: float = 0.0,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)

        self.TSPD_1 = TSPD(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, tri_reduction, tri_view_drop_p)
        self.TSPD_2 = TSPD(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, tri_reduction, tri_view_drop_p)
        self.TSPD_3 = TSPD(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, tri_reduction, tri_view_drop_p)
        self.TSPD_4 = TSPD(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, tri_reduction, tri_view_drop_p)

        self.TSPP_4 = TSPP(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, tri_reduction, tri_view_drop_p)
        self.TSPP_3 = TSPP(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, tri_reduction, tri_view_drop_p)
        self.TSPP_2 = TSPP(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, tri_reduction, tri_view_drop_p)
        self.TSPP_1 = TSPP(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, tri_reduction, tri_view_drop_p, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)

        x1 = self.TSPD_1(x0)
        x2 = self.TSPD_2(x1)
        x3 = self.TSPD_3(x2)
        x4 = self.TSPD_4(x3)

        u4 = self.TSPP_4(x4, x3)
        u3 = self.TSPP_3(u4, x2)
        u2 = self.TSPP_2(u3, x1)
        u1 = self.TSPP_1(u2, x0)

        logits = self.final_conv(u1)
        return logits
