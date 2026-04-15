# DCA-MMD
# Multi-Scale Dual Cross-Attentive MMD for Few-Shot Image Classification


## Overview

<img width="554" height="346" alt="image" src="https://github.com/user-attachments/assets/23bf346a-b47e-4678-adb6-599c7ac27fdf" />

## Code Prerequisites

The following packages are required to run the scripts:

- [PyTorch >= version 1.4](https://pytorch.org)

- [tensorboard](https://www.tensorflow.org/tensorboard)

Some comparing methods may require additional packages to run (e.g, OpenCV in DeepEMD and qpth, cvxpy in MetaOptNet).

## Dataset prepare

The dataset should be placed in dir "./data/dataset_name" with the same format. 

The miniimagenet and tieredimagenet-DeepEMD dataset can be downloaded from [FRN](https://drive.google.com/drive/folders/1gHt-Ynku6Yc3mz6aKVTppIfNmzML1sNG). The CIFAR-FS and FC100 datasets can be downloaded from [DeepEMD](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing).

## Train and Test

We follow the pretrain method from [FewTURE](https://github.com/mrkshllr/FewTURE) for Swin-Tiny and ViT-Small backbone, [MCL](https://github.com/cyvius96/prototypical-network-pytorch) for ResNet-12 backbone.

Download the pretrain weights from [Google Drive](https://drive.google.com/drive/folders/1Y2mEmOQHcTcKprVlZbtvRgXsPXT7IiD2?usp=drive_link) and extract it into the `pretrain/` folder.

Moreover, The train/test config and saved checkpoints are saved in the following format as above.

Download the meta-train snapshot from [Google Drive](https://drive.google.com/drive/folders/1CGkmW7rayh5sFjwjgE2w8t4XOValfbLi?usp=drive_link) and extract it into the `snapshots/` folder.

### Train and Test

For example,to train 5-way 1-shot on miniimagenet GPU 0
```
python experiments/run_trainer.py   
--cfg configs_DCAMMD/miniImagenet/DCAMMD_linear_triplet_N5K1.yaml   
-pt pretrain/ResNet/mini   
-d 0   
model.forward_encoding FCN_R12   
model.mmd.AMMD_feature 1
```
For example,to test 5-way 5-shot on FC100 GPU 0
```
python experiments/run_evaluator.py   
--cfg configs_DCAMMD/FC100/DCAMMD_linear_triplet_N5K5_R12.yaml   
-c checkpoint/FC100/masked_ratio_0.0/FC100_DCAMMD_linear_triplet_N5K5/ebest_5way_5shot.pth   
-d 0   
model.forward_encoding FCN_R12   
model.mmd.AMMD_feature 1
```

## Few-shot Classification Results
<img width="668" height="513" alt="image" src="https://github.com/user-attachments/assets/f25fed77-d2b4-4aa9-a1b8-46062c54fde9" />
<img width="670" height="430" alt="image" src="https://github.com/user-attachments/assets/99d9b8f8-56b2-4c25-ac01-43d8d7aee06a" />



## Contact

If you encounter any issues or have questions about using the code, feel free to contact me.
