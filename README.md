# [HTD: Heterogeneous Task Decoupling for Two-stage Object Detection](https://ieeexplore.ieee.org/document/9615001)

## 2022/03/08/ News!
Welcome to follow our new works [SIGMA](https://github.com/CityU-AIM-Group/SIGMA) (CVPR'22) and [SCAN](https://github.com/CityU-AIM-Group/SCAN) (AAAI'22 ORAL), which establish pixel-level graphs on the anchor-free object detector FCOS with some delicate designs.


## Environment

mmcv-full: 1.2.1
mmdet: 2.7
torch: 1.6.0
cudatoolkit: 10.0


## Get start
This work is based on the mmdetection framework. Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.
Please follow the official instruction of [mmdetection](https://mmdetection.readthedocs.io/en/latest/) and use our config files (configs/htd/...) to run the code.

## COCO test-dev evaluation

The submitted results for the official evaluation are based on ResNet-101-DCN ([Rank 50](https://competitions.codalab.org/competitions/20794#results))
![image](https://github.com/CityU-AIM-Group/HTD/blob/main/coco.png)

## Well-trained models 


Well-trained models (ResNet-50, ResNet-101, ResNet-101-DCN) are available at this [onedrive link](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/EhB8tbV-4e5PiAG8kNVDsXUBnNPcsUmbue53CGMM3yIuPw?e=fvsrL2)

## Contact
If you have any problems, please feel free to contact me at wuyangli2-c@my.cityu.edu.hk

## Abstract
Decoupling the sibling head has recently shown great potential in relieving the inherent task-misalignment problem in two-stage object detectors. However, existing works design similar structures for the classification and regression, ignoring task-specific characteristics and feature demands. Besides, the shared knowledge that may benefit the two branches is neglected, leading to potential excessive decoupling and semantic inconsistency. To address these two issues, we propose Heterogeneous task decoupling (HTD) framework for object detection, which utilizes a Progressive Graph (PGraph) module and a Border-aware Adaptation (BA) module for task-decoupling. Specifically, we first devise a Semantic Feature Aggregation (SFA) module to aggregate global semantics with image-level supervision, serving as the shared knowledge for the task-decoupled framework. Then, the PGraph module performs progressive graph reasoning, including local spatial aggregation and global semantic interaction, to enhance semantic representations of region proposals for classification. The proposed BA module integrates multi-level features adaptively, focusing on the low-level border activation to obtain representations with spatial and border perception for regression. Finally, we utilize the aggregated knowledge from SFA to keep the instance-level semantic consistency (ISC) of decoupled frameworks. Extensive experiments demonstrate that HTD outperforms existing detection works by a large margin, and achieves single-model 50.4%AP and 33.2% AP s on COCO test-dev set using ResNet-101-DCN backbone, which is the best entry among state-of-the-arts under the same configuration. 

![image](https://github.com/CityU-AIM-Group/HTD/blob/main/overall.png)




