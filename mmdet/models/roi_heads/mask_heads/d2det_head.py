import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init, normal_init

from mmdet.models.builder import HEADS, build_loss

from mmdet.core import mask_target
def see(data):
    print('max: ', torch.max(data))
    print('mean: ', torch.mean(data))
    print('min: ', torch.min(data), '\n')

@HEADS.register_module
class D2DetHead(nn.Module):

    def __init__(self,
                 num_convs=8,
                 roi_feat_size=14,
                 in_channels=256,
                 num_classes=80,
                 conv_kernel_size=3,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=36),
                 MASK_ON=False):
        super(D2DetHead, self).__init__()
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = 576
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.MASK_ON = MASK_ON
        if isinstance(norm_cfg, dict) and norm_cfg['type'] == 'GN':
            assert self.conv_out_channels % norm_cfg['num_groups'] == 0

        self.convs = []
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            stride = 2 if i == 0 else 1
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=False))
        self.convs = nn.Sequential(*self.convs)

        self.D2Det_reg = nn.Conv2d(self.conv_out_channels, 4, 3, padding=1)
        self.D2Det_mask = nn.Conv2d(self.conv_out_channels, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def print_(self, *key):

        print(key)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # TODO: compare mode = "fan_in" or "fan_out"
                kaiming_init(m)
        normal_init(self.D2Det_reg, std=0.001)
        normal_init(self.D2Det_mask, std=0.001)

    def forward(self, x, idx=None):
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        x0 = self.convs(x)
        x_m = self.D2Det_mask(x0)
        x_r = self.relu(self.D2Det_reg(x0))
        return x_r, x_m

    def get_target(self, sampling_results):
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results],
                               dim=0).cpu()
        pos_gt_bboxes = torch.cat(
            [res.pos_gt_bboxes for res in sampling_results], dim=0).cpu()
        assert pos_bboxes.shape == pos_gt_bboxes.shape
        num_rois = pos_bboxes.shape[0]
        map_size = 7
        targets = torch.zeros((num_rois, 4, map_size, map_size), dtype=torch.float)
        points = torch.zeros((num_rois, 4, map_size, map_size), dtype=torch.float)
        masks = torch.zeros((num_rois, 1, map_size, map_size), dtype=torch.float)

        for j in range(map_size):
            y = pos_bboxes[:, 1] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / map_size * (j + 0.5)

            dy = (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / (map_size - 1)
            for i in range(map_size):
                x = pos_bboxes[:, 0] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / map_size * (i + 0.5)

                dx = (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / (map_size - 1)

                targets[:, 0, j, i] = x - pos_gt_bboxes[:, 0]
                targets[:, 1, j, i] = pos_gt_bboxes[:, 2] - x
                targets[:, 2, j, i] = y - pos_gt_bboxes[:, 1]
                targets[:, 3, j, i] = pos_gt_bboxes[:, 3] - y

                idx = ((x - pos_gt_bboxes[:, 0] >= dx) & (pos_gt_bboxes[:, 2] - x >= dx) & (
                            y - pos_gt_bboxes[:, 1] >= dy) & (pos_gt_bboxes[:, 3] - y >= dy))

                masks[idx, 0, j, i] = 1

                points[:, 0, j, i] = x
                points[:, 1, j, i] = y
                points[:, 2, j, i] = pos_bboxes[:, 2] - pos_bboxes[:, 0]
                points[:, 3, j, i] = pos_bboxes[:, 3] - pos_bboxes[:, 1]

        targets = targets.cuda()
        points = points.cuda()
        masks = masks.cuda()


        return points, targets, masks

    def get_bboxes_avg(self, det_bboxes, D2Det_pred, D2Det_pred_mask, img_meta):
        # TODO: refactoring
        assert det_bboxes.shape[0] == D2Det_pred.shape[0]
        det_bboxes = det_bboxes
        D2Det_pred = D2Det_pred
        cls_scores = det_bboxes[:, [4]]
        det_bboxes = det_bboxes[:, :4]
        map_size = 7
        targets = torch.zeros((det_bboxes.shape[0], 4, map_size, map_size), dtype=torch.float, device=D2Det_pred.device)
        idx = (torch.arange(0, map_size).float() + 0.5).cuda() / map_size
        h = (det_bboxes[:, 3] - det_bboxes[:, 1]).view(-1, 1, 1)
        w = (det_bboxes[:, 2] - det_bboxes[:, 0]).view(-1, 1, 1)
        y = det_bboxes[:, 1].view(-1, 1, 1) + h * idx.view(1, map_size, 1)
        x = det_bboxes[:, 0].view(-1, 1, 1) + w * idx.view(1, 1, map_size)

        targets[:, 0, :, :] = x - D2Det_pred[:, 0, :, :] * w
        targets[:, 2, :, :] = x + D2Det_pred[:, 1, :, :] * w
        targets[:, 1, :, :] = y - D2Det_pred[:, 2, :, :] * h
        targets[:, 3, :, :] = y + D2Det_pred[:, 3, :, :] * h

        targets = targets.permute(0, 2, 3, 1).view(targets.shape[0], -1, 4)
        ious = (D2Det_pred_mask.view(-1, map_size * map_size, 1) > 0.0).float()

        targets = torch.sum(targets * ious, dim=1) / (torch.sum(ious, dim=1) + 0.00001)

        aa = torch.isnan(targets)
        if aa.sum() != 0:
            print('nan error...')

        bbox_res = torch.cat([targets, cls_scores], dim=1)
        bbox_res[:, [0, 2]].clamp_(min=0, max=img_meta[0]['img_shape'][1] - 1)
        bbox_res[:, [1, 3]].clamp_(min=0, max=img_meta[0]['img_shape'][0] - 1)

        return bbox_res

