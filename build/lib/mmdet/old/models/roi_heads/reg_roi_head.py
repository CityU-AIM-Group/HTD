import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class RegRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(RegRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))
        self.glbctx_head = build_head(
            dict(
            type='GlobalContextHead',
            num_ins=5,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=81,
            loss_weight=3.0))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

        self.glbctx_head.init_weights()

    def _fuse_glbctx(self, roi_feats, glbctx_feat, rois):
        """Fuse global context feats with roi feats."""
        assert roi_feats.size(0) == rois.size(0)
        img_inds = torch.unique(rois[:, 0].cpu(), sorted=True).long()
        fused_feats = torch.zeros_like(roi_feats)
        for img_id in img_inds:
            inds = (rois[:, 0] == img_id.item())
            fused_feats[inds] = roi_feats[inds] + glbctx_feat[img_id]

        return fused_feats
    def _bbox_forward(self, stage, x, rois, glbctx_feat, sampling_results=None,img_metas=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[0]
        bbox_roi_extractor_reg = self.bbox_roi_extractor[1]

        if stage == 0:
            bbox_head = self.bbox_head[stage]
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            bbox_feats = self._fuse_glbctx(bbox_feats, glbctx_feat, rois)
            cls_score, bbox_pred = bbox_head(bbox_feats)
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        elif stage == 1:
            if sampling_results:
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                b1_ = rois[...,0] == 0
                b2_ = rois[...,0] == 1
                b1 = pos_rois[...,0] == 0
                b2 = pos_rois[...,0] == 1
                num_pos_1 = torch.sum(b1)
                num_pos_2 = torch.sum(b2)
                num_boxs_1 = torch.sum(b1_)

                bbox_head = self.bbox_head[stage]
                bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                                rois)
                bbox_feats = self._fuse_glbctx(bbox_feats, glbctx_feat, rois)

                bbox_feats_reg = bbox_roi_extractor_reg(x[:bbox_roi_extractor.num_inputs],
                                                pos_rois)
                cls_score, bbox_pred = bbox_head(bbox_feats,bbox_feats_reg, x[:bbox_roi_extractor.num_inputs], rois, self.bbox_head[0].fc_cls)
                bbox_pred_new = cls_score.new_zeros(cls_score.size(0), 4)
                bbox_pred_new[:num_pos_1] = bbox_pred[:num_pos_1]
                bbox_pred_new[num_boxs_1:num_boxs_1+num_pos_2] = bbox_pred[num_pos_1:]
                bbox_results = dict(
                    cls_score=cls_score, bbox_pred=bbox_pred_new, bbox_feats=bbox_feats)
            else: #testing
                bbox_head = self.bbox_head[stage]
                bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                                rois)

                bbox_feats = self._fuse_glbctx(bbox_feats, glbctx_feat, rois)

                bbox_feats_reg = bbox_roi_extractor_reg(x[:bbox_roi_extractor.num_inputs],
                                                rois)
                cls_score, bbox_pred = bbox_head(bbox_feats, bbox_feats_reg, x[:bbox_roi_extractor.num_inputs], rois, self.bbox_head[0].fc_cls)
                bbox_results = dict(
                    cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg, img_metas, glbctx_feat):

        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois, glbctx_feat, sampling_results, img_metas)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)

        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)
        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        rcnn_train_cfg = self.train_cfg[0]
        lw = self.stage_loss_weights[0]
        # assign gts and sample proposals
        sampling_results = []

        bbox_assigner = self.bbox_assigner[0]
        bbox_sampler = self.bbox_sampler[0]
        num_imgs = len(img_metas)

        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        for j in range(num_imgs):
            assign_result = bbox_assigner.assign(
                proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                gt_labels[j])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[j],
                gt_bboxes[j],
                gt_labels[j],
                feats=[lvl_feat[j][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        mc_pred, glbctx_feat = self.glbctx_head(x)
        loss_glbctx = self.glbctx_head.loss(mc_pred, gt_labels)
        losses['loss_glbctx'] = loss_glbctx

        # bbox head forward and loss
        bbox_results = self._bbox_forward_train(0, x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                rcnn_train_cfg, img_metas, glbctx_feat)

        for name, value in bbox_results['loss_bbox'].items():
            losses[f's{0}.{name}'] = (
                value * lw if 'loss' in name else value)
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        roi_labels = bbox_results['bbox_targets'][0]
        with torch.no_grad():
            roi_labels = torch.where(
                roi_labels == self.bbox_head[0].num_classes,
                bbox_results['cls_score'][:, :-1].argmax(1),
                roi_labels)
            proposal_list = self.bbox_head[0].refine_bboxes(
                bbox_results['rois'], roi_labels,
                bbox_results['bbox_pred'], pos_is_gts, img_metas)

        #--------------------------------Graph Reasoning--------------------------------------------
        rcnn_train_cfg = self.train_cfg[1]
        lw = self.stage_loss_weights[1]
        sampling_results = []
        bbox_assigner = self.bbox_assigner[1]
        bbox_sampler = self.bbox_sampler[1]
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        for j in range(num_imgs):
            assign_result = bbox_assigner.assign(
                proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                gt_labels[j])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[j],
                gt_bboxes[j],
                gt_labels[j],
                feats=[lvl_feat[j][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        bbox_results = self._bbox_forward_train(1, x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                rcnn_train_cfg, img_metas, glbctx_feat)
        for name, value in bbox_results['loss_bbox'].items():
            losses[f's{1}.{name}'] = (
                value * lw if 'loss' in name else value)
        return losses
    def _random_jitter(self, sampling_results, img_metas, amplitude=0.15):
        """Ramdom jitter positive proposals for training."""
        for sampling_result, img_meta in zip(sampling_results, img_metas):
            bboxes = sampling_result.pos_bboxes
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            max_shape = img_meta['img_shape']
            if max_shape is not None:
                new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)

            sampling_result.pos_bboxes = new_bboxes
        return sampling_results
    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        rois = bbox2roi(proposal_list)
        mc_pred, glbctx_feat = self.glbctx_head(x)

        bbox_results = self._bbox_forward(0, x, rois, glbctx_feat)
        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(
            len(proposals) for proposals in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        ms_scores.append(cls_score)
         # refinging bboxes for the second stage
        bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
        rois = torch.cat([
            self.bbox_head[0].regress_by_class(rois[j], bbox_label[j],
                                               bbox_pred[j],
                                               img_metas[j])
            for j in range(num_imgs)
        ])

        #--------------------------------Graph Reasoning--------------------------------------------
        bbox_results = self._bbox_forward(1, x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(
            len(proposals) for proposals in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        ms_scores.append(cls_score)
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]
        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        return bbox_results
