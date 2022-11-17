# PARNet

This repository provides the PyTorch source code of the PARNet proposed in the following paper:

**[Mining Discriminative Food Regions for Accurate Food Recognition](https://bmvc2019.org/wp-content/uploads/papers/0839-paper.pdf)**
<br>
Jianing Qiu, Frank P.-W. Lo, Yingnan Sun, Siyao Wang, Benny Lo
<br>
**Spotlight** at BMVC 2019

![](assets/parnet_animation.gif)

## Requirements
- opencv-python
- pytorch>=0.4.1
- matplotlib
- scikit-image

The code was initially written in pytorch 0.4.1, but has been tested recently and can run with pytorch 1.13.0 and CUDA 11.6 on ubuntu 20.04 as well.

## Dataset
Please use the link provided in the **data** folder to download the Sushi-50 dataset proposed in this paper, and put it in the **data** folder afterwards. For Food-101 and Vireo-172, please follow their respective instruction for downloading, and put them in the **data** folder as well.

## Citation
If you find this code useful for your research, please consider citing:
```
@inProceedings{qiu2019mining,
  title={Mining Discriminative Food Regions for Accurate Food Recognition},
  author={Qiu, Jianing and Lo, Frank Po Wen and Sun, Yingnan and Wang, Siyao and Lo, Benny},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2019}
}
```
## Acknowledgements
This work is supported by the Innovative Passive Dietary Monitoring Project funded by the Bill & Melinda Gates Foundation (Opportunity ID: OPP1171395).
