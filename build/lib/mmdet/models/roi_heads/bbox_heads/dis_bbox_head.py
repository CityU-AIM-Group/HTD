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


@HEADS.register_module()
class DisBBoxHead(BBoxHead):
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
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DisBBoxHead, self).__init__(*args, **kwargs)
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
        self.conv_out_channels = 1024
        self.fc_out_channels = 1024
        self.relu = nn.ReLU(inplace=True)

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.fc_reg = nn.Linear(self.conv_out_channels, 4*self.num_classes)
        self.convs = []
        self.middle_channel = 256
        self.in_channels = 256
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

            elif i == self.num_reg_convs - 1:
                self.convs.append(
                    ConvModule(
                        self.middle_channel,
                        1024,
                        self.conv_kernel_size,
                        stride=stride,
                        padding=padding,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=None,
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


    def init_weights(self):
        super(DisBBoxHead, self).init_weights()
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)
        for m in self.fcs.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):

        x_reg = self.convs(x_reg)
        x_reg = self.avg_pool(x_reg)
        x_reg = x_reg.view(x_reg.size(0), -1)

        x_cls = x_cls.flatten(1)
        x_cls = self.fcs(x_cls)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None


        return cls_score, bbox_pred
