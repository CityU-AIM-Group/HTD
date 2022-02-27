import mmcv
import numpy as np

from . import CocoDataset
from .builder import DATASETS
from .custom import CustomDataset
import pickle as pkl

@DATASETS.register_module()
class DHD_Traffic(CocoDataset):
    CLASSES = ('Pedestrian','Cyclist','Car','Truck','Van')