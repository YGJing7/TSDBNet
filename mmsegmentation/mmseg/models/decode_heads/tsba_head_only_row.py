import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from ..backbones.mamba_vision import MambaVisionMixer


@MODELS.register_module()
class TSBAHead(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 in_index=[0, 1, 2, 3],
                 input_transform='multiple_select',
                 depth_context=(2, 1),
                 mixer_cfg=None,
                 dropout_ratio=0.1,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            in_index=in_index,
            input_transform=input_transform,
            align_corners=False,
            dropout_ratio=dropout_ratio,
            **kwargs)

        self.mixer_cfg = mixer_cfg or dict(d_state=16, d_conv=4, expand=2)

        """ Unified Channel """
        c1, c2, c3, c4 = self.in_channels[:4]
        self.lateral_1 = ConvBNAct(c1, channels, k=1, p=0)  # 1/4
        self.lateral_2 = ConvBNAct(c2, channels, k=1, p=0)  # 1/8
        self.lateral_3 = ConvBNAct(c3, channels, k=1, p=0)  # 1/16
        self.lateral_4 = ConvBNAct(c4, channels, k=1, p=0)  # 1/32        

        """ TriScan Mamba Context Path """
        d3, d4 = depth_context
        self.triscan_context_4 = nn.Sequential(*[TriScanBlock(channels, self.mixer_cfg, dropout_ratio) for _ in range(max(1, d4))])
        self.triscan_context_3 = nn.Sequential(*[TriScanBlock(channels, self.mixer_cfg, dropout_ratio) for _ in range(max(1, d3))])
        self.dsup_4to3 = DSUp(channels, channels, kernel_size=3, activation='relu')
        self.lgam_low = LGAM(F_g=channels, F_l=channels, F_int=max(8, channels // 2), kernel_size=3, groups=max(1, channels // 2))
        self.fuse_34 = ConvBNAct(channels, channels, k=3)
        
        """ Upsampling """
        self.dsup_3to2 = DSUp(channels, channels, kernel_size=3, activation='relu')

        """ Boundary Aware Refinement Path """
        self.barm2 = BARM(channels)
        self.barm1 = BARM(channels)
        self.dsup_2to1 = DSUp(channels, channels, kernel_size=3, activation='relu')
        self.lgam_high = LGAM(F_g=channels, F_l=channels, F_int=max(8, channels // 2), kernel_size=3, groups=max(1, channels // 2))
        self.fuse_12 = ConvBNAct(channels, channels, k=3)

        """ gate_attention """
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def _forward_context(self, p3, p4):
        c4 = self.triscan_context_4(p4)
        c4_up = self.dsup_4to3(c4, size=p3.shape[-2:])
        p3_gate = self.lgam_low(g=c4_up, x=p3)
        fused = self.fuse_34(c4_up + p3_gate)
        c3 = self.triscan_context_3(fused)
        return c3

    def _forward_boundary(self, p1, p2, p_deep):
        c2 = self.barm2(p2, p_deep)
        c2up = self.dsup_2to1(c2)
        p1_gate = self.lgam_high(g=c2up, x=p1)
        fused = self.fuse_12(c2up + p1_gate)
        c1 = self.barm1(p1, fused)
        return c1
    
    def _gate_fusion(self, c3, c1):
        ctx  = F.interpolate(c3, size=c1.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        bnd = F.interpolate(c1, size=c1.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        g = self.gate(torch.cat([ctx, bnd], dim=1))
        fused = g * ctx + (1.0 - g) * bnd
        return fused

    def forward(self, inputs):
        feats = self._transform_inputs(inputs)
        p1 = self.lateral_1(feats[0])
        p2 = self.lateral_2(feats[1])
        p3 = self.lateral_3(feats[2])
        p4 = self.lateral_4(feats[3])

        c3 = self._forward_context(p3, p4)
        p_deep = self.dsup_3to2(c3)
        c1 = self._forward_boundary(p1, p2, p_deep)
        fused = self._gate_fusion(c3, c1)

        out = self.cls_seg(fused)
        out = F.interpolate(out, size=inputs[0].shape[-2:], mode='bilinear', align_corners=self.align_corners)
        return out


def act_layer(name='relu', inplace=True, neg_slope=0.2):
    name = (name or 'relu').lower()
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    if name == 'relu6':
        return nn.ReLU6(inplace=inplace)
    if name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=neg_slope, inplace=inplace)
    if name == 'gelu':
        return nn.GELU()
    if name == 'hswish':
        return nn.Hardswish(inplace=inplace)
    raise NotImplementedError(f'Unknown activation: {name}')


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, bias=False, act='relu'):
        if p is None: p = k // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=bias),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(act_layer(act))
        super().__init__(*layers)


class DSUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dw = ConvBNAct(in_channels, in_channels, k=kernel_size, g=in_channels, act=activation)
        self.pw = ConvBNAct(in_channels, out_channels, k=1, p=0, act=None)
    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode='nearest')
            return self.pw(self.dw(x))
        x = self.up(x)
        return self.pw(self.dw(x))


class LGAM(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super().__init__()
        pad = kernel_size // 2
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, padding=pad, groups=max(1, min(groups, F_g)), bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, padding=pad, groups=max(1, min(groups, F_l)), bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.act = act_layer(activation)
    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        att = self.psi(self.act(self.W_g(g) + self.W_x(x)))
        return x * att


class CAB(nn.Module):
    def __init__(self, channels, reduction=16, activation='relu'):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            act_layer(activation),
            nn.Conv2d(hidden, channels, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        w = self.sigmoid(avg + mx)
        return w


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.cat([avg, mx], dim=1)
        w = self.sigmoid(self.bn(self.conv(w)))
        return w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, activation='relu'):
        super().__init__()
        self.cab = CAB(channels, reduction=reduction, activation=activation)
        self.sab = SAB(kernel_size=7)
    def forward(self, x):
        x = x * self.cab(x)
        x = x * self.sab(x)
        return x

class TriScanBlock(nn.Module):
    def __init__(self, channels, mixer_cfg=None, dropout=0.0):
        super().__init__()
        assert MambaVisionMixer is not None, "MambaVisionMixer is required by TriScanBlock."
        mixer_cfg = mixer_cfg or {}
        self.row_mixer  = MambaVisionMixer(d_model=channels, **mixer_cfg)
        self.col_mixer  = MambaVisionMixer(d_model=channels, **mixer_cfg)
        self.diag_mixer = MambaVisionMixer(d_model=channels, **mixer_cfg)
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _scan_rows(x, mixer):
        B, C, H, W = x.shape
        t = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        y = mixer(t)
        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return y

    @staticmethod
    def _scan_cols(x, mixer):
        B, C, H, W = x.shape
        t = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
        y = mixer(t)
        y = y.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        return y

    @staticmethod
    def _shift_rows_for_diag(x, inverse=False):
        """ Approximate diagonal scanning by cyclically shifting rows """
        B, C, H, W = x.shape
        x_bhcw = x.permute(0, 2, 1, 3).contiguous()  # [B, H, C, W]
        device = x.device
        w = torch.arange(W, device=device).unsqueeze(0).expand(H, W)   # [H, W]
        r = torch.arange(H, device=device).unsqueeze(1).expand(H, W)   # [H, W]
        idx = (w + r) % W if inverse else (w - r) % W
        idx = idx.view(1, H, 1, W).expand(B, H, x_bhcw.size(2), W)
        x_shift = torch.gather(x_bhcw, dim=-1, index=idx)
        return x_shift.permute(0, 2, 1, 3).contiguous()

    def _scan_diag(self, x):
        xs = self._shift_rows_for_diag(x, inverse=False)
        ys = self._scan_rows(xs, self.diag_mixer)
        y  = self._shift_rows_for_diag(ys, inverse=True)
        return y

    def forward(self, x):
        y_row = self._scan_rows(x, self.row_mixer)
        # y_col = self._scan_cols(x, self.col_mixer)
        # y_dia = self._scan_diag(x)
        # y = torch.cat([y_row, y_col], dim=1)  # [B, 3C, H, W]
        # y = self.fuse(y)
        y = self.drop(y_row)
        out = x + y
        return out


class BARM(nn.Module):
    def __init__(self, in_channels):
        super(BARM, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)

    def forward(self, x, pred):
        residual = x
        pred = self.conv(pred)
        pred = torch.sigmoid(pred)
        
        #reverse attention
        background_att = 1 - pred
        background_x= x * background_att
        
        #boudary attention
        edge_pred = make_laplace(pred, 1)  
        pred_feature = x * edge_pred

        fusion_feature = torch.cat([background_x, pred_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff