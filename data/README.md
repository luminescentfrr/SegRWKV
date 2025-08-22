Download dataset [here](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing)
Download pretrained vrwkv [here](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_b_in1k_224.pth)

Please organize the dataset as follows:

```
data/
├── nnUNet_raw/
│   ├── Dataset702_AbdomenMR/
│   │   ├── imagesTr
│   │   │   ├── amos_0507_0000.nii.gz
│   │   │   ├── amos_0508_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── amos_0507.nii.gz
│   │   │   ├── amos_0508.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json
│   │
│   │
│   ├── ...
├── pretrained/
│   ├── vrwkv_s_in1k_224.pth
```
