# WGAN-QC
PyTorch implementation of [Wasserstein GAN With Quadratic Transport Cost](http://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Wasserstein_GAN_With_Quadratic_Transport_Cost_ICCV_2019_paper.html).
The code is build under Python 3.6, PyTorch 1.0
# Usage
DATASET: CelebA/CelebA-HQ/LSUN <br>
DATASET_PATH: path to the root folder of a dataset <br>
OUTPUT_ROOT: the root path of your output <br>
GPU_ID: gpu id
1. Install dependency: [CVXOPT](https://cvxopt.org/) for linear programming. <br> 
2. Download your dataset, unzip it and put it in DATASET_PATH. <br>

On the CelebA dataset, run 
```
python wgan_qc_resnet2.py --dataset celeba --dataroot DATASET_PATH --output_root OUTPUT_ROOT --output_dir celeba_results --batchSize 64 --imageSize 64 --Giters 60000 --gamma 0.1 --EMA_startIter 55000 --gpu_ids GPU_ID
```

On the CelebA-HQ dataset, run
```
python wgan_qc_resnet1.py --dataset celebahq --dataroot DATASET_PATH --output_root OUTPUT_ROOT --output_dir celebahq_results --batchSize 16 --imageSize 256 --Giters 125000 --gamma 0.05 --EMA_startIter 120000 --gpu_ids GPU_ID
```

On the LSUN dataset, run
```
python wgan_qc_resnet1.py --dataset lsun --dataroot DATASET_PATH --output_root OUTPUT_ROOT --output_dir lsun_results --batchSize 64 --imageSize 64 --Giters 150000 --gamma 0.1 --EMA_startIter 145000 --gpu_ids GPU_ID
```

You are welcome to cite our work using:
```
@InProceedings{Liu_2019_ICCV,
author = {Liu, Huidong and Gu, Xianfeng and Samaras, Dimitris},
title = {Wasserstein GAN With Quadratic Transport Cost},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```