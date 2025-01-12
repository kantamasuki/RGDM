# Image data generation with Renormalization Group-based Diffusion Model (RGDM)

This is the implementation of the RGDM and DDPM used in numerical experiments of protein structure prediction in [Generative Diffusion Model with Inverse Renormalization Group Flows](https://arxiv.org/abs/). Part of the codes is based on [the original work of DDPM](https://arxiv.org/abs/2006.11239) by J. Ho et al. while we are responsible for any issues regarding the codes provided here.


## Version Information 
We use `venv` virtual environment. The environment and the versions of the Python libraries used in our work are as follows:

- GPU: NVIDIA RTX 6000 Ada Generation
    - CUDA Version: 12.2
- Python: 3.10.12
- PyTorch: 2.2.0
    - torch-dct: 0.1.6
- clean-fid: 0.1.35

Any reasonably recent version will be fine. We note that the clean-fid library (see [here](https://github.com/GaParmar/clean-fid) for details) is necessary for evaluating the performance of diffusion models. 

## Dataset
In this work, we used [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets. We note that the resized images are created from FFHQ images with 128x128 resolution by `./data/ffhq_dataset/resize_ffhq.py`.

## Sampling
Firstly, all the trained models (i.e., checkpoints) used in the paper are provided in `./ckpt_paper/`. Also, the configuration files during the training are provided in `./RGDM_configs` and `./DDPM_configs`. The sampling in our work can be reproduced with these models and configuration files. For example, to sample FFHQ64 images with the RGDM trained at $T=200$ (or, CIFAR10 images with the DDPM trained at $T=300$) run the following:
```
python RGDM_sample.py --config RGDM_configs/RGDM_FFHQ64_T200.json --ckpt ckpt_paper/RGDM_FFHQ64_T200.pt
python DDPM_sample.py --config DDPM_configs/DDPM_CIFAR10_T300.json --ckpt ckpt_paper/DDPM_CIFAR10_T300.pt
```
Sampled images will be saved in `./sample`.

## Evaluation
One can evaluate the performance of the model (the RGDM or DDPM) trained with a configuration file by
```
python RGDM_evaluate.py --config PATH_TO_CONFIG --ckpt PATH_TO_MODEL --num_images N
python DDPM_evaluate.py --config PATH_TO_CONFIG --ckpt PATH_TO_MODEL --num_images N
```
In the numerical experiments, we took $N=50000$. For example, to evaluate the diffusion models described above, run the following:
```
python RGDM_evaluate.py --config RGDM_configs/RGDM_FFHQ64_T200.json --ckpt ckpt_paper/RGDM_FFHQ64_T200.pt --num_images 50000
python DDPM_evaluate.py --config DDPM_configs/DDPM_CIFAR10_T300.json --ckpt ckpt_paper/DDPM_CIFAR10_T300.pt --num_images 50000
```

## Training
To reproduce the training with FFHQ images of resolution 64x64, please unzip `./data/ffhq_dataset/ffhq64.tar.gz` as `./data/ffhq_dataset/ffhq64`. To train the RGDM or DDPM, run the following:
```
python RGDM_train.py --config PATH_TO_CONFIG
python DDPM_train.py --config PATH_TO_CONFIG
```
These create a new log directory in `./logs/`. The configuration and checkpoints during the training will be saved there.
