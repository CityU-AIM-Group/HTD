import mmcv
import numpy as np

from . import CocoDataset
from .builder import DATASETS
from .custom import CustomDataset
import pickle as pkl

@DATASETS.register_module()
class VOCDataset_coco(CocoDataset):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')