# # import torch.nn as nn
# # from mmcv.cnn import ConvModule
# # from mmcv.runner import auto_fp16, force_fp32

# # from mmdet.models.builder import HEADS
# # from mmdet.models.utils import ResLayer, SimplifiedBasicBlock

# # from mmcv.cnn.utils import constant_init, kaiming_init
# # @HEADS.register_module()
# # class GlobalContextHead(nn.Module):
# #     """Global context head."""

# #     def __init__(self,
# #                  num_ins,
# #                  num_convs=4,
# #                  in_channels=256,
# #                  conv_out_channels=256,
# #                  num_classes=81,
# #                  loss_weight=1.0,
# #                  conv_cfg=None,
# #                  norm_cfg=None,
# #                  conv_to_res=False):
# #         super(GlobalContextHead, self).__init__()
# #         self.num_ins = num_ins
# #         self.num_convs = num_convs
# #         self.in_channels = in_channels
# #         self.conv_out_channels = conv_out_channels
# #         self.num_classes = num_classes
# #         self.loss_weight = loss_weight
# #         self.conv_cfg = conv_cfg
# #         self.norm_cfg = norm_cfg
# #         self.conv_to_res = conv_to_res
# #         self.fp16_enabled = False


# #         if self.conv_to_res:
# #             num_res_blocks = num_convs // 2
# #             self.convs = ResLayer(
# #                 SimplifiedBasicBlock,
# #                 in_channels,
# #                 self.conv_out_channels,
# #                 num_res_blocks,
# #                 conv_cfg=self.conv_cfg,
# #                 norm_cfg=self.norm_cfg)
# #             self.num_convs = num_res_blocks
# #         else:
# #             self.convs = nn.ModuleList()
# #             for i in range(self.num_convs):
# #                 in_channels = self.in_channels if i == 0 else conv_out_channels
# #                 self.convs.append(
# #                     ConvModule(
# #                         in_channels,
# #                         conv_out_channels,
# #                         3,
# #                         padding=1,
# #                         conv_cfg=self.conv_cfg,
# #                         norm_cfg=self.norm_cfg))

# #         self.adaptative_conv =  nn.Conv2d(256, 256, 1, 1)
# #         self.pool = nn.AdaptiveAvgPool2d(1)
# #         self.pool77 = nn.AdaptiveAvgPool2d(7)
# #         self.fc = nn.Linear(conv_out_channels, num_classes)

# #         self.criterion = nn.BCEWithLogitsLoss()
# #         # self.gmp_pooling = nn.AdaptiveMaxPool2d(1)
# #     def init_weights(self):
# #         nn.init.normal_(self.fc.weight, 0, 0.01)
# #         nn.init.constant_(self.fc.bias, 0)
# #         kaiming_init(self.adaptative_conv)

# #     @auto_fp16()
# #     def forward(self, feats):
# #         x = feats[-1]
# #         for i in range(self.num_convs):
# #             x = self.convs[i](x)

# #         out_feat = self.pool77(self.adaptative_conv(x))
# #         x = self.pool(x)
# #         # x2 = self.gmp_pooling(x)
# #         # x = x1 + x2

# #         # multi-class prediction
# #         mc_pred = x.reshape(x.size(0), -1)
# #         mc_pred = self.fc(mc_pred)


# #         return mc_pred, out_feat

# #     @force_fp32(apply_to=('pred', ))
# #     def loss(self, pred, labels):
# #         labels = [lbl.unique() for lbl in labels]
# #         targets = pred.new_zeros(pred.size())
# #         for i, label in enumerate(labels):
# #             targets[i, label] = 1.0
# #         loss = self.loss_weight * self.criterion(pred, targets)
# #         return loss
# import torch.nn as nn
# from mmcv.cnn import ConvModule
# from mmcv.runner import auto_fp16, force_fp32

# from mmdet.models.builder import HEADS
# from mmdet.models.utils import ResLayer, SimplifiedBasicBlock


# @HEADS.register_module()
# class GlobalContextHead(nn.Module):
#     """Global context head."""

#     def __init__(self,
#                  num_ins,
#                  num_convs=4,
#                  in_channels=256,
#                  conv_out_channels=256,
#                  num_classes=81,
#                  loss_weight=1.0,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  conv_to_res=False):
#         super(GlobalContextHead, self).__init__()
#         self.num_ins = num_ins
#         self.num_convs = num_convs
#         self.in_channels = in_channels
#         self.conv_out_channels = conv_out_channels
#         self.num_classes = num_classes
#         self.loss_weight = loss_weight
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.conv_to_res = conv_to_res
#         self.fp16_enabled = False

#         if self.conv_to_res:
#             num_res_blocks = num_convs // 2
#             self.convs = ResLayer(
#                 SimplifiedBasicBlock,
#                 in_channels,
#                 self.conv_out_channels,
#                 num_res_blocks,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg)
#             self.num_convs = num_res_blocks
#         else:
#             self.convs = nn.ModuleList()
#             for i in range(self.num_convs):
#                 in_channels = self.in_channels if i == 0 else conv_out_channels
#                 self.convs.append(
#                     ConvModule(
#                         in_channels,
#                         conv_out_channels,
#                         3,
#                         padding=1,
#                         conv_cfg=self.conv_cfg,
#                         norm_cfg=self.norm_cfg))

#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(conv_out_channels, num_classes)

#         self.criterion = nn.BCEWithLogitsLoss()

#     def init_weights(self):
#         nn.init.normal_(self.fc.weight, 0, 0.01)
#         nn.init.constant_(self.fc.bias, 0)

#     @auto_fp16()
#     def forward(self, feats):
#         x = feats[-1]
#         x = self.conv_to_res(x)
#         # import matplotlib.pyplot as plt
#         # import torch
#         # feat = torch.mean(x,dim=1).squeeze().cpu().numpy()
#         # import cv2
#         # # from skimage import transform,data
#         # for i, feat in enumerate(feats):
#         #     # print(feat.shape)
#         #     feat = torch.mean(feat,dim=1).squeeze().cpu().numpy()
#         #     h = feat.shape[0]*4
#         #     w = feat.shape[1]*4
#         #     new = cv2.resize(feat, (w,h))
#         #     new = cv2.flip(new,1)
#         #     plt.imsave('/home/wuyang/Dropbox/lab/people2_{}.png'.format(i),new)

#         # for i in range(self.num_convs):
#         #     x = self.convs[i](x)

#         # feat = torch.mean(x,dim=1).squeeze().cpu().numpy()
#         # h = feat.shape[0]*4
#         # w = feat.shape[1]*4

#         # new = cv2.resize(feat, (w,h))
#         # new = cv2.flip(new,1)

#         # plt.imsave('/home/wuyang/Dropbox/lab/feat_conv.png',new)

#         x = self.pool(x)
#         # feat = x.squeeze().cpu().numpy()

#         # plt.plot(range(len(list(feat))), feat, color='red')
#         # plt.xlabel('channel')
#         # plt.savefig('/home/wuyang/Dropbox/lab/dis.png')







#         mc_pred = x.reshape(x.size(0), -1)
#         mc_pred = self.fc(mc_pred)

#         return mc_pred, x

#     @force_fp32(apply_to=('pred', ))
#     def loss(self, pred, labels):
#         labels = [lbl.unique() for lbl in labels]
#         targets = pred.new_zeros(pred.size())
#         for i, label in enumerate(labels):
#             targets[i, label] = 1.0
#         loss = self.loss_weight * self.criterion(pred, targets)
#         return loss
# import torch.nn as nn
# from mmcv.cnn import ConvModule
# from mmcv.runner import auto_fp16, force_fp32

# from mmdet.models.builder import HEADS
# from mmdet.models.utils import ResLayer, SimplifiedBasicBlock

# from mmcv.cnn.utils import constant_init, kaiming_init
# @HEADS.register_module()
# class GlobalContextHead(nn.Module):
#     """Global context head."""

#     def __init__(self,
#                  num_ins,
#                  num_convs=4,
#                  in_channels=256,
#                  conv_out_channels=256,
#                  num_classes=81,
#                  loss_weight=1.0,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  conv_to_res=False):
#         super(GlobalContextHead, self).__init__()
#         self.num_ins = num_ins
#         self.num_convs = num_convs
#         self.in_channels = in_channels
#         self.conv_out_channels = conv_out_channels
#         self.num_classes = num_classes
#         self.loss_weight = loss_weight
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.conv_to_res = conv_to_res
#         self.fp16_enabled = False


#         if self.conv_to_res:
#             num_res_blocks = num_convs // 2
#             self.convs = ResLayer(
#                 SimplifiedBasicBlock,
#                 in_channels,
#                 self.conv_out_channels,
#                 num_res_blocks,
#                 conv_cfg=self.conv_cfg,
#                 norm_cfg=self.norm_cfg)
#             self.num_convs = num_res_blocks
#         else:
#             self.convs = nn.ModuleList()
#             for i in range(self.num_convs):
#                 in_channels = self.in_channels if i == 0 else conv_out_channels
#                 self.convs.append(
#                     ConvModule(
#                         in_channels,
#                         conv_out_channels,
#                         3,
#                         padding=1,
#                         conv_cfg=self.conv_cfg,
#                         norm_cfg=self.norm_cfg))

#         self.adaptative_conv =  nn.Conv2d(256, 256, 1, 1)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.pool77 = nn.AdaptiveAvgPool2d(7)
#         self.fc = nn.Linear(conv_out_channels, num_classes)

#         self.criterion = nn.BCEWithLogitsLoss()
#         # self.gmp_pooling = nn.AdaptiveMaxPool2d(1)
#     def init_weights(self):
#         nn.init.normal_(self.fc.weight, 0, 0.01)
#         nn.init.constant_(self.fc.bias, 0)
#         kaiming_init(self.adaptative_conv)

#     @auto_fp16()
#     def forward(self, feats):
#         x = feats[-1]
#         for i in range(self.num_convs):
#             x = self.convs[i](x)

#         out_feat = self.pool77(self.adaptative_conv(x))
#         x = self.pool(x)
#         # x2 = self.gmp_pooling(x)
#         # x = x1 + x2

#         # multi-class prediction
#         mc_pred = x.reshape(x.size(0), -1)
#         mc_pred = self.fc(mc_pred)


#         return mc_pred, out_feat

#     @force_fp32(apply_to=('pred', ))
#     def loss(self, pred, labels):
#         labels = [lbl.unique() for lbl in labels]
#         targets = pred.new_zeros(pred.size())
#         for i, label in enumerate(labels):
#             targets[i, label] = 1.0
#         loss = self.loss_weight * self.criterion(pred, targets)
#         return loss
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock


@HEADS.register_module()
class GlobalContextHead(nn.Module):
    """Global context head."""

    def __init__(self,
                 num_ins,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=81,
                 loss_weight=1.0,
                 conv_cfg=None,
                 norm_cfg=None,
                 conv_to_res=False):
        super(GlobalContextHead, self).__init__()
        self.num_ins = num_ins
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.conv_to_res = conv_to_res
        self.fp16_enabled = False

        if self.conv_to_res:
            num_res_blocks = num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                in_channels,
                self.conv_out_channels,
                num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.num_convs = num_res_blocks
        else:
            self.convs = nn.ModuleList()
            for i in range(self.num_convs):
                in_channels = self.in_channels if i == 0 else conv_out_channels
                self.convs.append(
                    ConvModule(
                        in_channels,
                        conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv_out_channels, num_classes)

        self.criterion = nn.BCEWithLogitsLoss()

    def init_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    @auto_fp16()
    def forward(self, feats):
        x = feats[-1]
        for i in range(self.num_convs):
            x = self.convs[i](x)
        x = self.pool(x)

        # multi-class prediction
        mc_pred = x.reshape(x.size(0), -1)
        mc_pred = self.fc(mc_pred)

        return mc_pred, x

    @force_fp32(apply_to=('pred', ))
    def loss(self, pred, labels):
        labels = [lbl.unique() for lbl in labels]
        targets = pred.new_zeros(pred.size())
        for i, label in enumerate(labels):
            targets[i, label] = 1.0
        loss = self.loss_weight * self.criterion(pred, targets)
        return loss