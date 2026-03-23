# EVLF: Early Vision-Language Fusion for Generative Dataset Distillation

Official implementation of our CVPR 2026 paper.

This repository contains the training and inference code for EVLF.

## Repository Structure

- `Train/`: training code for the fusion module
- `Inference/`: sampling, DiT training, and evaluation code
- `environment.yml`: Conda environment definition

## Getting Started

This guide explains how to set up the environment, configure dataset paths, and run both training and inference for this project.

### 1. Create and Activate the Conda Environment

Run the following commands in the project root:

```bash
conda create -f environment.yml -n your_env_name
conda activate your_env_name
```

### 2. Configure the ImageIDC Dataset Path

Before running the code, you need to modify the dataset path in two scripts.

#### 2.1 Update the Training Script

Edit `Train/train_script/train_idc.sh`

Replace the line:

```bash
ORIGINAL_PATH=xxx
```

with the path to your local ImageIDC dataset.

#### 2.2 Update the Inference Script

Edit `Inference/scripts/idc.sh`

Replace the line:

```bash
IMAGENET_FOLDER=xxx
```

with the path to your local ImageIDC dataset.

### 3. Run Training and Inference

#### 3.1 Grant Execution Permission to the Inference Script

```bash
chmod +x Inference/scripts/idc.sh
```

#### 3.2 Run the Training Script

```bash
sh Train/train_script/train_idc.sh
```

#### 3.3 Run the Inference Script

```bash
sh Inference/scripts/idc.sh
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{cai2026evlf,
  title={{EVLF}: Early Vision-Language Fusion for Generative Dataset Distillation},
  author={Cai, Wenqi and Zou, Yawen and Li, Guang and Gu, Chunzhi and Zhang, Chao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
