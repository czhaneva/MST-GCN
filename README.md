# MST-GCN
This is the official implemntation for "Multi-scale spatial temporal graph convolutional network for skeleton-based action recognition" AAAI-2021 ([pdf](https://ojs.aaai.org/index.php/AAAI/article/view/16197/16004))

## Requirements
  ![Python >=3.6.7](https://img.shields.io/badge/Python->=3.6.7-yellow.svg)    ![PyTorch >=1.2.0](https://img.shields.io/badge/PyTorch->=1.2.0-blue.svg)     ![CUDA >=10.0.130](https://img.shields.io/badge/CUDA->=10.0.130-blue.svg)

## Data Preparation
### NTU RGB+D dataset
- Download the raw data of [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
- Preprocess the data with `python data_gen/ntu_gendata.py`
- Generate the bone data with `python data_gen/gen_bone_data.py`
- Generate the motion data with `python data_gen/gen_motion_data.py`
### Kinetics-Skeleton
- Download the data from [GoogleDrive](https://drive.google.com/drive/folders/1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ) provided by [st-gcn](https://github.com/yysijie/st-gcn)
- Preprocess the data with `python data_gen/kinetics_gendata.py`
- Generate the bone data with `python data_gen/gen_bone_data.py`
- Generate the motion data with `python data_gen/gen_motion_data.py`

## Training-Testing
Change the config file depending on what you want.

```bash
# train on NTU RGB+D xview joint stream
$ sh run.sh 0,1,2,3 4 2022 0
# or
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=2022 main.py --config config/ntu/train_joint_amstgcn_ntu.yaml
```

## Citation
Please cite our paper if you find this repository useful in your resesarch:

```
@inproceedings{chen2021multi,
  title={Multi-scale spatial temporal graph convolutional network for skeleton-based action recognition},
  author={Chen, Zhan and Li, Sicheng and Yang, Bing and Li, Qinghan and Liu, Hong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={2},
  pages={1113--1122},
  year={2021}
}
```

## Acknowledgement
The framework of our code is extended from the following repositories. We sincerely thank the authors for releasing the codes.
- The framework of our code is based on [st-gcn](https://github.com/yysijie/st-gcn) and [2s-agcn](https://github.com/lshiwjx/2s-AGCN).

## Licence

This project is licensed under the terms of the MIT license.

## Contact
For any questions, feel free to contact: `zhanchen_cz@pku.edu.cn` or `czchenzhan@gmail.com`
