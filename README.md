# 3rd place solution for xView2 Damage Assessment Challenge

Eugene Khvedchenya, February 2020

This repository contains source code for my solution to [xView2 challenge](https://xview2.com). My solution was scored second (0.803) on public LB and third (0.805) on private hold-out dataset.

# Approach in a nutshell

- Ensemble of semantic segmentation models. 
- Trained with weighted CE to address class imbalance.
- Heavy augmentations to prevent over-fitting and increase robustness to misalignment of pre- and post- images. 
- Shared encoder for pre- and post- images. Extracted feature are concatenated and sent to decoder. 
- Bunch of encoders (ResNets, Densenets, EfficientNets) and two decoders: Unet and FPN.
- 1 round of Pseudolabeling
- Ensemble using weighted averaging. Weights optimized for every model on corresponding validation data.

# Training

- Install dependencies from `requirements.txt`
- Follow `train.sh` 

# Inference

For inference using pre-trained models please download full archive from Releases tab and run `predict_37_weighted.py` script.

# License

MIT