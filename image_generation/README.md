# Image data generation with Renormalization Group-based Diffusion Model (RGDM)

This is the implementation of the RGDM and DDPM in image generation in [Generative Diffusion Model with Inverse Renormalization Group Flows](https://arxiv.org/abs/). Part of the codes is based on [the original work of DDPM](https://arxiv.org/abs/2006.11239) by J. Ho et al.


## Version Information 
We use `venv` virtual environment. The environment and the versions of GPU and Python libraries used in our work are as follows, while any reasonably recent version will be fine: 

- GPU: NVIDIA RTX 6000 Ada Generation
    - CUDA Version: 12.2
- Python: 3.10.12
- PyTorch: 2.2.0
    - torch-dct: 0.1.6
- clean-fid: 0.1.35

We note that the [clean-fid library](https://github.com/GaParmar/clean-fid) is necessary for evaluating the performance of the models. 

## Dataset
In our work, we used [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets. We note that we resized the FFHQ images with resolution 128x128 to 64x64 by `./data/ffhq_dataset/resize_ffhq.py`. We provide resized images [here]() (`image_paper_results/ffhq64.tar.gz`). 

## Paper results
All the trained models (i.e., checkpoints) used in the paper are provided [here](). The configuration files of the training are provided in `./RGDM_configs` and `./DDPM_configs`. To reproduce the results in the paper, download `paper_results/image_paper_resutls/results` (i.e., make directory `./results`). 

## Sampling
To perform the sampling, run `MODEL_sample.py`, specifying the configuration file and checkpoint. For example, to sample FFHQ64 images with the RGDM trained at $T=200$ (or, CIFAR10 images with the DDPM trained at $T=300$) run the following:
```
python RGDM_sample.py --config RGDM_configs/RGDM_FFHQ64_T200.json --ckpt results/RGDM_FFHQ64_T200.pt
python DDPM_sample.py --config DDPM_configs/DDPM_CIFAR10_T300.json --ckpt results/DDPM_CIFAR10_T300.pt
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
python RGDM_evaluate.py --config RGDM_configs/RGDM_FFHQ64_T200.json --ckpt results/RGDM_FFHQ64_T200.pt --num_images 50000
python DDPM_evaluate.py --config DDPM_configs/DDPM_CIFAR10_T300.json --ckpt results/DDPM_CIFAR10_T300.pt --num_images 50000
```

## Training
To reproduce the training with FFHQ images of resolution 64x64, please unzip `ffhq64.tar.gz` provided [here]() below `./data/ffhq_dataset` (This will make directory `./data/ffhq_dataset/ffhq64`). To train the RGDM or DDPM, run the following:
```
python RGDM_train.py --config PATH_TO_CONFIG
python DDPM_train.py --config PATH_TO_CONFIG
```
These create a new log directory `./logs/...`, in which the configuration and checkpoints during the training will be saved.
