import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class finalRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 with_global = False,
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
        self.with_global = with_global
        super(finalRoIHead, self).__init__(
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


        if self.with_global:
            self.global_head = build_head(
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
        if self.with_global:
            self.global_head.init_weights()

    def _fuse_global(self, roi_feats, global_feat, rois):
        """Fuse global context feats with roi feats."""
        assert roi_feats.size(0) == rois.size(0)
        img_inds = torch.unique(rois[:, 0].cpu(), sorted=True).long()
        fused_feats = torch.zeros_like(roi_feats)
        for img_id in img_inds:
            inds = (rois[:, 0] == img_id.item())
            fused_feats[inds] = roi_feats[inds] + global_feat[img_id]

        return fused_feats
    def _bbox_forward(self, stage, x, rois, global_feat=None, sampling_results=None,img_metas=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[0]
        roi_extractor_lvl0 = self.bbox_roi_extractor[1]

        if stage == 0:
            bbox_head = self.bbox_head[stage]
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            if self.with_global:
                bbox_feats = self._fuse_global(bbox_feats, global_feat, rois)
            cls_score, bbox_pred = bbox_head(bbox_feats)
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        elif stage == 1:
            if sampling_results:
                # training mode
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                b1_ = rois[...,0] == 0
                b2_ = rois[...,0] == 1
                b1 = pos_rois[...,0] == 0
                b2 = pos_rois[...,0] == 1
                num_pos_1 = torch.sum(b1)
                num_pos_2 = torch.sum(b2)
                num_boxs_1 = torch.sum(b1_)
                bbox_head = self.bbox_head[stage]
                bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
                bbox_feats_reg = roi_extractor_lvl0(x[:bbox_roi_extractor.num_inputs], pos_rois)






                if self.with_global:
                    cls_score, bbox_pred = bbox_head(bbox_feats, bbox_feats_reg, x[:bbox_roi_extractor.num_inputs], rois, self.bbox_head[0].fc_cls, pos_rois, global_feat)
                else:
                    cls_score, bbox_pred = bbox_head(bbox_feats, bbox_feats_reg, x[:bbox_roi_extractor.num_inputs], rois, self.bbox_head[0].fc_cls)
                bbox_pred_pos_ = cls_score.new_zeros(cls_score.size(0), 4)
                bbox_pred_pos_[:num_pos_1] = bbox_pred[:num_pos_1]
                bbox_pred_pos_[num_boxs_1:num_boxs_1+num_pos_2] = bbox_pred[num_pos_1:]
                bbox_results = dict(
                    cls_score=cls_score, bbox_pred=bbox_pred_pos_)
            else:
                # testing mode
                bbox_head = self.bbox_head[stage]
                bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
                bbox_feats_reg = roi_extractor_lvl0(x[:bbox_roi_extractor.num_inputs], rois)
                # if self.with_global:
                #     bbox_feats = self._fuse_global(bbox_feats, global_feat, rois)
                if self.with_global:
                    cls_score, bbox_pred = bbox_head(bbox_feats, bbox_feats_reg, x[:bbox_roi_extractor.num_inputs], rois, self.bbox_head[0].fc_cls, rois, global_feat)
                else:
                    cls_score, bbox_pred = bbox_head(bbox_feats, bbox_feats_reg, x[:bbox_roi_extractor.num_inputs], rois, self.bbox_head[0].fc_cls)

                bbox_results = dict(
                    cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg, img_metas, global_feat=None):

        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois, global_feat, sampling_results, img_metas)
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
        if self.with_global:
            mc_pred, global_feat = self.global_head(x)
            loss_glbctx = self.global_head.loss(mc_pred, gt_labels)
            losses['loss_global'] = loss_glbctx
        else:
            global_feat = None

        #--------------------------------Stage 1: Common Head--------------------------------------------

        bbox_results = self._bbox_forward_train(0, x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                rcnn_train_cfg, img_metas, global_feat)

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

        #--------------------------------Stage 2: Graph Reasoning--------------------------------------------
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
                                                rcnn_train_cfg, img_metas, global_feat)
        for name, value in bbox_results['loss_bbox'].items():
            losses[f's{1}.{name}'] = (
                value * lw if 'loss' in name else value)
        return losses

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

        if self.with_global:
            mc_pred, global_feat = self.global_head(x)
        else:
            global_feat = None



        #--------------------------------Stage 1: Common Head--------------------------------------------
        bbox_results = self._bbox_forward(0, x, rois, global_feat)
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
        #--------------------------------Stage 2: Graph Reasoning--------------------------------------------
        bbox_results = self._bbox_forward(1, x, rois, global_feat)
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
    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []
            if self.with_global:
                mc_pred, global_feat = self.global_head(x)
            else:
                global_feat = None
            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois ,global_feat)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(
                        dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[[]
                                for _ in range(self.mask_head[-1].num_classes)]
                               ]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]
