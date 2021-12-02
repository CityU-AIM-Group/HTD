from mmcv.cnn.bricks import build_plugin_layer
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor
from mmcv.cnn.bricks import  ConvModule
import torch
@ROI_EXTRACTORS.register_module()
class AdptRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels.

    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 aggregation='sum',
                 pre_cfg=None,
                 post_cfg=None,
                 edge=2,
                 **kwargs):
        super(AdptRoIExtractor, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        self.edge = edge
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
        self.att = torch.nn.Sequential(
            self.pool,
            self.conv1,
            torch.nn.Tanh(),
            self.conv2
        )
        # build pre/post processing modules

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # mark the starting channels for concat mode

        roi_feat = []
        atts = []
        for i in range(num_levels):
            roi_feat_t = self.roi_layers[i](feats[i], rois)
            atts.append(self.att(roi_feat_t).squeeze().unsqueeze(0))
            roi_feat.append(roi_feat_t.unsqueeze(0))


        roi_feat = torch.cat(roi_feat, dim=0)
        lvl, n, c, x, y = roi_feat.size()

        atts = torch.cat(atts,dim=0).softmax(0)

        assert atts.size(0) == lvl
        assert atts.size(1) == n
        atts = atts.unsqueeze(-1).repeat(1, 1, c * x * y ).view(lvl, n, c, x, y)

        roi_feats = (atts * roi_feat).sum(0)
        roi_feats_enhance = self.roi_layers[0](feats[0], rois)
        roi_feats_enhance[:, :, self.edge :-self.edge , self.edge :-self.edge] = 0


        return roi_feats+roi_feats_enhance
