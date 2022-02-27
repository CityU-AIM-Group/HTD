from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from .base import BaseDetector
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, xavier_init
@DETECTORS.register_module()
class HDGraphDet(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(HDGraphDet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.fix_channel = 1024
        self.num_cls = roi_head.bbox_head.num_classes
        self.feature_fix = ConvModule(2048, self.fix_channel, 3, 1, norm_cfg=dict(type='BN'))
        self.cmprs = ConvModule(self.fix_channel, 256, 3, 1, norm_cfg=dict(type='BN'))
        self.gap_pooling = nn.AdaptiveAvgPool2d(1)
        self.gmp_pooling = nn.AdaptiveMaxPool2d(1)
        self.roi_poolng = nn.AdaptiveMaxPool2d((7, 7))
        self.image_cls = nn.Linear(self.fix_channel, self.num_cls)
        # self.sigmoid = nn.Sigmoid()

        self.global_loss = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True))

        # self.global_loss = nn.BCELoss()
        self.relu = torch.nn.ReLU(inplace=True)

        self.init_weights(pretrained=pretrained)


    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            fpn = self.neck(x)
        return fpn, x[-1]
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, C5 = self.extract_feat(img)
        image_ = self.feature_fix(C5)
        gap = self.gap_pooling(image_)
        gmp = self.gmp_pooling(image_)
        image_pred = self.image_cls((gap + gmp).squeeze())
        image_label = []
        # prototype = torch.cat((self.image_cls.weight,
        #                        self.image_cls.bias.unsqueeze(1)), 1).detach()
        # prototype = self.prototype_mapping(prototype.t()).t()
        for bs in gt_labels:
            label = image_pred.new_zeros(self.num_cls)
            label[bs.tolist()] = 1
            image_label.append(label)

        image_label = torch.stack(image_label)
        losses = dict()
        loss_global = {}
        loss_global['global_loss'] = self.global_loss(image_pred, image_label)
        losses.update(loss_global)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # print(self.rpn_head.forward_train)
            # input()
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        global_feat = self.cmprs(image_)
        global_feat = self.roi_poolng(global_feat)
        roi_losses = self.roi_head.forward_train(x, global_feat, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x, C5 = self.extract_feat(img)
        image_ = self.feature_fix(C5)
        global_feat = self.cmprs(image_)
        global_feat = self.roi_poolng(global_feat)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, global_feat, proposal_list, img_metas, rescale=rescale)
