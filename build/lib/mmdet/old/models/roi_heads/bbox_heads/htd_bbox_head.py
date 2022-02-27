import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, xavier_init
import torch
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
import torch.nn.functional as F
from mmdet.models.backbones.resnet import Bottleneck
import matplotlib.pyplot as plt
import numpy as np
# import ipdb
from mmdet.models.losses import accuracy
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from mmdet.core import multi_apply
from mmcv.runner import auto_fp16, force_fp32
from torch.optim import Adam

def see( data):
    print('max: ', torch.max(data))
    print('mean: ', torch.mean(data))
    print('min: ', torch.min(data), '\n')

@HEADS.register_module()
class HTDBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=2,
                 num_reg_convs=4,
                 num_reg_fcs=0,
                 alpha=1,
                 relpace = False,
                 average = False,
                 edge = 1,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=36),
                 *args,
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(HTDBBoxHead, self).__init__(*args, **kwargs)
        self.conv_kernel_size = 3
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.alpha = alpha
        self.relpace = relpace
        self.edge = edge
        self.average = average
        self.conv_out_channels = 1024
        self.fc_out_channels = 1024
        self.relu = nn.ReLU(inplace=True)
        self.gcn_in = 1024
        self.gcn_out = 1024
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.fc_reg = nn.Linear(self.conv_out_channels  , 4)
        self.convs = []
        self.middle_channel = 16 * 36
        for i in range(self.num_reg_convs):
            stride = 1
            padding = (self.conv_kernel_size - 1) // 2
            if i == 0:
                self.convs.append(
                    ConvModule(
                        self.in_channels,
                        self.middle_channel,
                        self.conv_kernel_size,
                        stride=stride,
                        padding=padding,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=False))
            elif i == self.num_reg_convs-1:
                self.convs.append(
                ConvModule(
                    self.middle_channel,
                    1024,
                    self.conv_kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg = None,
                    bias=False))
            else:
                self.convs.append(
                ConvModule(
                    self.middle_channel,
                    self.middle_channel,
                    self.conv_kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=False))
        self.convs = nn.Sequential(*self.convs)
        self.fcs = []
        for i in range(self.num_cls_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            self.fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            self.fcs.append(self.relu)
        self.fcs = nn.Sequential(*self.fcs)
        self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        # self.project = nn.Linear(1025, 256)
        self.graph_lvl0_cls = nn.Linear(self.gcn_in, self.gcn_out)
        self.graph_lvl1_cls = nn.Linear(self.gcn_in, self.gcn_out)
        self.graph_lvl2_cls = nn.Linear(self.gcn_in, self.gcn_out)
        self.graph_lvl3_cls = nn.Linear(self.gcn_in, self.gcn_out)
        self.graph_layer_cls = [self.graph_lvl0_cls, self.graph_lvl1_cls, self.graph_lvl2_cls, self.graph_lvl3_cls]
    def map_roi_levels(self, rois, num_levels):
        finest_scale = 56
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls
    def init_weights(self):
        super(HTDBBoxHead, self).init_weights()
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)
        for m in self.fcs.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        for m in self.graph_layer_cls:
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def _fuse_global(self, roi_feats, glbctx_feat, rois):
        """Fuse global context feats with roi feats."""
        assert roi_feats.size(0) == rois.size(0)
        img_inds = torch.unique(rois[:, 0].cpu(), sorted=True).long()
        fused_feats = torch.zeros_like(roi_feats)
        for img_id in img_inds:
            inds = (rois[:, 0] == img_id.item())
            fused_feats[inds] = roi_feats[inds] + glbctx_feat[img_id]
        return fused_feats

    def forward(self, x_cls, x_reg, feat, rois, fc_cls_0, enhanced_feat=None, pos_rois=None, global_feat=None):
        prototype = torch.cat((fc_cls_0.weight,fc_cls_0.bias.unsqueeze(1)), 1).detach()
        bs = int(torch.max(rois[...,0]))+ 1

        if global_feat is not None:
            x_cls_glb = self._fuse_global(x_cls, global_feat, rois)
            x_reg = self._fuse_global(x_reg, global_feat, pos_rois)
            x_cls_glb =  self.fcs(x_cls_glb.flatten(1))

        if self.relpace:
            x_reg[:, :, :self.edge, :] = 0
            x_reg[:, :, -self.edge:, :] = 0
            x_reg[:, :, :, :self.edge] = 0
            x_reg[:, :, :, -self.edge:] = 0
            x_reg = x_reg + self.alpha * enhanced_feat
        elif self.average:
            x_reg[:, :, :self.edge, :] = x_reg[:, :, :self.edge, :] * 0.5
            x_reg[:, :, -self.edge:, :] = x_reg[:, :, -self.edge:, :] * 0.5
            x_reg[:, :, :, :self.edge] = x_reg[:, :, :, :self.edge] * 0.5
            x_reg[:, :, :, -self.edge:] = x_reg[:, :, :, -self.edge:] * 0.5
            x_reg[:, :, 0, 0] *= 2
            x_reg[:, :, 0, -1] *= 2
            x_reg[:, :, -1, 0] *= 2
            x_reg[:, :, -1, -1] *= 2
            x_reg += 0.5 * enhanced_feat

        else:
            x_reg = x_reg + self.alpha * enhanced_feat

        x_reg = self.convs(x_reg)

        x_reg = self.avg_pool(x_reg)
        x_reg = x_reg.view(x_reg.size(0), -1)
        # cls head
        x_cls = x_cls.flatten(1)
        x_cls = self.fcs(x_cls)

        sam = torch.mm(fc_cls_0(x_cls).softmax(-1),prototype)
        target_lvls = self.map_roi_levels(rois, len(feat))
        refined_feature_cls = x_cls.new_zeros(x_cls.size(0), self.gcn_out)
        t = 0; tp = x_cls.new_zeros(1,1024)
        for b in range(bs):
            bs_indx = rois[...,0]==b
            lvl_indx = [target_lvls == 0,target_lvls == 1, target_lvls == 2, target_lvls == 3]
            for i in range(len(feat)):
                bs_lvl_indx = torch.logical_and(lvl_indx[i], bs_indx)
                if bs_lvl_indx.any():
                    #classification
                    sam_ = sam[bs_lvl_indx,:]
                    rois_ = rois[bs_lvl_indx, 1:5]
                    h_local_mask = bbox_overlaps(rois_, rois_).fill_diagonal_(1.)
                    h_local_mask[h_local_mask > 0] = 1.
                    D = torch.diag(torch.sum(h_local_mask, dim=-1).pow(-0.5))
                    A_local = torch.mm(torch.mm(D, h_local_mask), D)
                    h_global_mask = (1. - h_local_mask)
                    roi_feat = x_cls[bs_lvl_indx, :]
                    roi_feat_mixed = torch.mm(A_local, roi_feat)
                    sim = torch.mm(sam_, sam_.t())
                    A_global = (h_global_mask * sim).softmax(-1)
                    new_cls = self.relu(self.graph_layer_cls[i](torch.matmul(A_global, roi_feat_mixed)))
                    refined_feature_cls[bs_lvl_indx] = new_cls
                else:
                    t += 0 * torch.sum(self.graph_layer_cls[i](tp))

        if global_feat is not None:
            feat_cls_new = x_cls_glb + refined_feature_cls
        else:
            feat_cls_new = x_cls + refined_feature_cls


        cls_score = self.fc_cls(feat_cls_new) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred

