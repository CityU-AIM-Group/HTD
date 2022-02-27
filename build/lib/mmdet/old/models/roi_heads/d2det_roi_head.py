import torch

from mmdet.core import bbox2result, bbox2roi, multiclass_nms1
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .standard_roi_head import StandardRoIHead
import numpy as np
import mmcv
import pycocotools.mask as mask_util
import torch.nn.functional as F
def see( data):
    print('max: ', torch.max(data))
    print('mean: ', torch.mean(data))
    print('min: ', torch.min(data), '\n')
@HEADS.register_module()
class D2DetRoIHead(StandardRoIHead):
    def __init__(self, reg_roi_extractor, d2det_head, **kwargs):
        assert d2det_head is not None
        super(D2DetRoIHead, self).__init__(**kwargs)
        self.reg_roi_extractor = build_roi_extractor(reg_roi_extractor)
        self.share_roi_extractor = False
        self.D2Det_head = build_head(d2det_head)
        self.loss_roi_reg = build_loss(dict(type='IoULoss', loss_weight=1.0))
        self.loss_roi_mask = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        self.MASK_ON = d2det_head.MASK_ON
        self.num_classes = d2det_head.num_classes

    def init_weights(self, pretrained):
        super(D2DetRoIHead, self).init_weights(pretrained)
        self.D2Det_head.init_weights()
        if not self.share_roi_extractor:
            self.reg_roi_extractor.init_weights()

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
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        bbox_results = self._bbox_forward_train(x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                img_metas)
        losses.update(bbox_results['loss_bbox'])
        return losses
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        bbox_results = super(D2DetRoIHead,
                             self)._bbox_forward_train(x, sampling_results,
                                                       gt_bboxes, gt_labels,
                                                       img_metas)
        # sampling_results = self._random_jitter(sampling_results, img_metas)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        reg_feats = self.reg_roi_extractor(
            x[:self.reg_roi_extractor.num_inputs], pos_rois)

        if reg_feats.shape[0] == 0:
            bbox_results['loss_bbox'].update(dict(loss_reg=reg_feats.sum() * 0, loss_mask=reg_feats.sum() * 0))
        else:
            reg_pred, reg_masks_pred = self.D2Det_head(reg_feats)
            reg_points, reg_targets, reg_masks = self.D2Det_head.get_target(sampling_results)
            reg_targets = reg_targets
            reg_points = reg_points
            reg_masks = reg_masks
            x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :] * reg_points[:, 2, :, :]
            x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :] * reg_points[:, 2, :, :]
            y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :] * reg_points[:, 3, :, :]
            y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :] * reg_points[:, 3, :, :]

            pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)

            x1_1 = reg_points[:, 0, :, :] - reg_targets[:, 0, :, :]
            x2_1 = reg_points[:, 0, :, :] + reg_targets[:, 1, :, :]
            y1_1 = reg_points[:, 1, :, :] - reg_targets[:, 2, :, :]
            y2_1 = reg_points[:, 1, :, :] + reg_targets[:, 3, :, :]

            pos_decoded_target_preds = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=1)

            loss_reg = self.loss_roi_reg(
                pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                pos_decoded_target_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                weight=reg_masks.reshape(-1))
            loss_mask = self.loss_roi_mask(
                reg_masks_pred.reshape(-1, reg_masks.shape[2] * reg_masks.shape[3]),
                reg_masks.reshape(-1, reg_masks.shape[2] * reg_masks.shape[3]))
            bbox_results['loss_bbox'].update(dict(loss_reg=loss_reg, loss_mask=loss_mask))
        return bbox_results
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=False)

        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]

        # print(det_bboxes.size())
        if det_bboxes.shape[0] != 0:
            reg_rois = bbox2roi([det_bboxes[:, :4]])
            reg_feats = self.reg_roi_extractor(
                x[:len(self.reg_roi_extractor.featmap_strides)], reg_rois)
            self.D2Det_head.test_mode = True
            reg_pred, reg_pred_mask = self.D2Det_head(reg_feats)
            det_bboxes = self.D2Det_head.get_bboxes_avg(det_bboxes,
                                                        reg_pred,
                                                        reg_pred_mask,
                                                        img_metas)
            # print(det_bboxes)
            # input()
            det_bboxes, det_labels = multiclass_nms1(det_bboxes[:, :4], det_bboxes[:, 4], det_labels,
                                                         self.num_classes, dict(type='soft_nms', iou_thr=0.5), 300)
            if rescale:
                scale_factor = det_bboxes.new_tensor(img_metas[0]['scale_factor'])
                det_bboxes[:, :4] /= scale_factor
        else:
            det_bboxes = torch.Tensor([])
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        return [bbox_results]

