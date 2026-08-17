import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@MODELS.register_module()
class EdgeHead(BaseDecodeHead):
    def __init__(self, 
                 in_channels,
                 channels,
                 num_classes,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            input_transform='multiple_select',
            **kwargs
        )
        # 统一通道
        c1, c2, c3, c4 = self.in_channels[:4]
        self.lateral_1 = ConvBNAct(c1, channels, k=1, p=0)
        self.lateral_2 = ConvBNAct(c2, channels, k=1, p=0)
        self.lateral_3 = ConvBNAct(c3, channels, k=1, p=0)
        self.lateral_4 = ConvBNAct(c4, channels, k=1, p=0)

        """ Conv Path """
        self.ccontext_4 = nn.Sequential(ConvBNAct(channels, channels, k=1), CBAM(channels))
        self.ccontext_3 = nn.Sequential(ConvBNAct(channels, channels, k=1), CBAM(channels))
        self.ccontext_2 = nn.Sequential(ConvBNAct(channels, channels, k=1), CBAM(channels))
        self.ccontext_1 = nn.Sequential(ConvBNAct(channels, channels, k=1), CBAM(channels))

        self.cfuse_34 = ConvBNAct(channels, channels, k=1)
        self.cfuse_23 = ConvBNAct(channels, channels, k=1)
        self.cfuse_12 = ConvBNAct(channels, channels, k=1)

        # 上采样 & 门控
        self.cdsup_4to3 = DSUp(channels, channels, kernel_size=3)
        self.clgam_3 = LGAM(channels, channels, F_int=max(8, channels // 2), kernel_size=3, groups=max(1, channels // 2))
        self.cdsup_3to2 = DSUp(channels, channels, kernel_size=3)
        self.clgam_2 = LGAM(channels, channels, F_int=max(8, channels // 2), kernel_size=3, groups=max(1, channels // 2))
        self.cdsup_2to1 = DSUp(channels, channels, kernel_size=3)
        self.clgam_b1 = LGAM(channels, channels, F_int=max(8, channels // 2), kernel_size=3, groups=max(1, channels // 2))


    def forward(self, inputs):
        feats = self._transform_inputs(inputs)
        p1 = self.lateral_1(feats[0])
        p2 = self.lateral_2(feats[1])
        p3 = self.lateral_3(feats[2])
        p4 = self.lateral_4(feats[3])

        c4 = self.ccontext_4(p4)
        c4_up = self.cdsup_4to3(c4, size=p3.shape[-2:])
        p3_gate = self.clgam_3(g=c4_up, x=p3)
        c3_in = self.cfuse_34(c4_up+p3_gate)

        c3 = self.ccontext_3(c3_in)
        c3_up = self.cdsup_3to2(c3, size=p2.shape[-2:])
        p2_gate = self.clgam_2(g=c3_up, x=p2)
        c2_in = self.cfuse_23(c3_up+p2_gate)

        c2 = self.ccontext_2(c2_in)
        c2_up = self.cdsup_2to1(c2, size=p1.shape[-2:])
        p1_gate = self.clgam_b1(g=c2_up, x=p1)
        c1_in = self.cfuse_12(c2_up+p1_gate)

        c1 = self.ccontext_1(c1_in)
        feat = F.interpolate(c1, inputs[0].shape[-2:], mode='bilinear', align_corners=self.align_corners)
        return self.cls_seg(feat)

    def loss(self, inputs, batch_data_samples, train_cfg) -> dict:
        """Forward function for training.
        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        edge_logits = self.forward(inputs)
        gt_edge = torch.stack(
            [ds.gt_edge_map.data for ds in batch_data_samples],
            dim=0
        )
        gt_edge = gt_edge.squeeze(1).long()
        gt_edge[gt_edge == self.ignore_index] = 0
        losses = self.loss_by_feat(edge_logits, gt_edge)
        return losses
    
    def loss_by_feat(self, edge_logits, gt_edge):
        """
        edge_logits: (N, 1, H, W)
        gt_edge: (N, H, W)
        """
        # resize prediction to GT size
        edge_logits = F.interpolate(
            input=edge_logits,
            size=gt_edge.shape[1:],
            mode='bilinear',
            align_corners=True
        )

        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    edge_logits,
                    gt_edge
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    edge_logits,
                    gt_edge
                )

        return loss


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, bias=False):
        if p is None: p = k // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*layers)

class DSUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dw = ConvBNAct(in_channels, in_channels, k=kernel_size, g=in_channels)
        self.pw = ConvBNAct(in_channels, out_channels, k=1, p=0)
    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode='nearest')
            return self.pw(self.dw(x))
        x = self.up(x)
        return self.pw(self.dw(x))

class LGAM(nn.Module):
    """
    Light Gated Attention Module
    """
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1):
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
        self.act = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        att = self.psi(self.act(self.W_g(g) + self.W_x(x)))
        return x * att


class CBAM(nn.Module):
    """CAB + SAB 的轻量组合：先通道再空间"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cab = CAB(channels, reduction=reduction)
        self.sab = SAB(kernel_size=7)
    def forward(self, x):
        x = x * self.cab(x)
        x = x * self.sab(x)
        return x

class CAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
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