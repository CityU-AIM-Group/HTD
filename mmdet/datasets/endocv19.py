import mmcv
import numpy as np

from . import CocoDataset
from .builder import DATASETS
from .custom import CustomDataset
import pickle as pkl

@DATASETS.register_module()
class EADDataset(CocoDataset):
    CLASSES = (
        'specularity',
        'saturation',
        'artifact',
        'blur',
        'contrast',
        'bubbles',
        'instrument',
    )