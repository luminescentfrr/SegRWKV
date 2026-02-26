# SegRWKV: Fast and Accurate Biomedical Image Segmentation via a Receptance-Weighted Key-Value Network

Official repository for: *[SegRWKV: Fast and Accurate Biomedical Image Segmentation via a Receptance-Weighted Key-Value Network]

![network](https://github.com/luminescentfrr/SegRWKV/blob/main/assets/URWKV.png)


## Installation

**Step-1:** Create a new conda environment & install requirements

```shell
conda create -n segrwkv python=3.10
conda activate segrwkv
```

**Step-2:** Install URWKV

```shell
git clone https://github.com/luminescentfrr/SegRWKV
cd SegRWKV/u_rwkv
pip install -e .
```

## Prepare data & pretrained model

**Dataset:**  

We use the same data & processing strategy following U-Mamba. Download dataset from [U-Mamba](https://github.com/bowang-lab/U-Mamba) and put them into the data folder. Then preprocess the dataset with following command:

```shell
export nnUNet_raw=./SegRWKV/data/nnUNet_raw
export nnUNet_preprocessed=./SegRWKV/data/nnUNet_preprocessed
export nnUNet_results=./SegRWKV/data/nnUNet_results


nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

**ImageNet pretrained model:** 

We use the ImageNet pretrained VisionRWKV-base and ResNetV2 model from [VRWKV](https://github.com/OpenGVLab/Vision-RWKV) and [ResNet](https://github.com/google-research/big_transfer/tree/master/bit_pytorch). You need to download the model checkpoint and put it into `data/pretrained/vrwkv_b_in1k_224.pth` and `data/pretrained/BiT-M-R50x1.npz`

```
wget https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_b_in1k_224.pth

wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz

```

## Training

Using the following command to train & evaluate SegRWKV

```shell
dos2unix scripts/train_AbdomenMR.sh
dos2unix scripts/train_Endoscopy.sh
dos2unix scripts/train_Microscopy.sh


# AbdomenMR dataset
bash scripts/train_AbdomenMR.sh MODEL_NAME
# Endoscopy dataset
bash scripts/train_Endoscopy.sh MODEL_NAME
# Microscopy dataset 
bash scripts/train_Microscopy.sh MODEL_NAME
```

Here  `MODEL_NAME` can be:

- `nnUNetTrainerURWKV`: URWKV model with ImageNet pretraining
- `nnUNetTrainerURWKVScratch`: URWKV model without ImageNet pretraining

You can download our model checkpoints [here]().

## Acknowledgements

We thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [RWKV](https://github.com/BlinkDL/RWKV-LM), [UMamba](https://github.com/bowang-lab/U-Mamba), [VRWKV](https://github.com/OpenGVLab/Vision-RWKV),  [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba) and [TransUNet](https://github.com/Beckschen/TransUNet) for making their valuable code & data publicly available.

## Citation

```

```
