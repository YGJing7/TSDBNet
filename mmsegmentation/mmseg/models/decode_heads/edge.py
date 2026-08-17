import cv2 as cv
import numpy as np
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class Edge(BaseDecodeHead):
    def __init__(self,in_channels,**kwargs):
        super(Edge, self).__init__(in_channels,input_transform = 'multiple_select',**kwargs)

        self.align1 = ConvModule(
            in_channels[3],
            in_channels[2],
            1)
        self.align2 = ConvModule(
            in_channels[2],
            in_channels[1],
            1)        
        self.align3 = ConvModule(
            in_channels[1],
            in_channels[0],
            1)

    def forward(self, inputs):
        inputs = [resize(
                level,
                size=inputs[0].shape[-2:],
                mode='bilinear'
            ) for level in inputs]        

        #stage4
        y3 = inputs[3]
        #stage3
        conv_y1 = self.align1(y3)
        y2 = inputs[2]+conv_y1
        #stage3
        conv_y2 = self.align2(y2)
        y1 = inputs[1]+conv_y2           
        #stage3
        conv_y3 = self.align3(y1)
        y0 = inputs[0]+conv_y3
        return self.cls_seg(y0)
    
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