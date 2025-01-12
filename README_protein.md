# Protein structure prediction with RGDM

This is the implementation of the RGDM and DDPM used in numerical experiments of protein structure prediction in [Generative Diffusion Model with Inverse Renormalization Group Flows](https://arxiv.org/abs/). Part of the codes is based on a previous work (['EigenFold' by B.Jing](https://github.com/bjing2016/EigenFold)) while we are responsible for any issues regarding the codes provided here.


## Version Information
We use Anaconda virtual environment. The environment and the versions of the Python libraries used in our work are as follows: 

- GPU: NVIDIA GeForce RTX 3090
    - CUDA Version: 11.7
- Python: 3.10.11
- PyTorch: 2.0.1
    - torch-cluster: 1.6.3
    - torch-dct: 0.1.6
    - torch_geometric: 2.5.0
    - torch-scatter: 2.1.2
    - torch-sparse: 0.6.18
    - torch-spline-conv: 1.2.2
- Biopython: 1.83
- e3nn: 0.5.1

Any reasonably recent version will be fine.

## Installation
(This installation part is largely based on [github.com/bjing2016/EigenFold](https://github.com/bjing2016/EigenFold).) 
To run our codes, install Pytorch and necessary libraries as
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-dct
pip install e3nn biopython
```

Firstly, one needs to download protein structures from the PDB by
```
bash download_pdb.sh ./data
python unpack_pdb.py --num_workers [N]
python make_splits.py
```
This will make `splits/limit256.csv`.

Also, to create amino-sequence features, which are used in the training and generation steps of the diffusion models, one needs to install [OmegaFold](https://github.com/bjing2016/OmegaFold) by:
```
wget https://helixon.s3.amazonaws.com/release1.pt
git clone https://github.com/bjing2016/OmegaFold
pip install --no-deps -e OmegaFold
```

Then, run the following to create necessary amino-sequence features by
```
python make_embeddings.py --out_dir ./embeddings --splits CSV_FILE --out_dir OUTPUT_DIRECTORY
```
For example, to perform the training (sampling), run the above with `--splits splits/limit256.csv` (`--splits splits/cameo2022.csv`). We note that these require approximately 1.1TB of storage in your computer in total. Also, these takes about a few days (or about one weak) with a single GPU.

Lastly, to evaluate the quality of sampled protein structures, compile `TMscore.cpp` and add the binary to your `PATH` with the name `TMscore`. For example, run 
```
g++ TMscore.cpp -o TMscore
mv TMscore /usr/local/bin/
```

## Sampling
Firstly, all the trained models (i.e., checkpoints) used in the paper are provided in `./workdir`; the RGDM is `./workdir/RGDM_trained/epoch_9.pt`, and the DDPM is `./workdir/DDPM_trained/epoch_4.pt`. Also, the sampled protein structures used in the paper are provided in `./workdir/RGDM_trained/samples` and `./workdir/DDPM_trained/samples`, respectively. To perform the sampling, run the following:
```
python RGDM_sample.py --model_dir workdir/RGDM_trained --ckpt epoch_9.pt --split_key ANY_KEY_OK --splits splits/cameo2022.csv --embeddings_key name
python DDPM_sample.py --model_dir workdir/DDPM_trained --ckpt epoch_4.pt --split_key ANY_KEY_OK --splits splits/cameo2022.csv --embeddings_key name
```
These create a new sample directory below `./RGDM_trained/` and `./DDPM_trained/`, respectively. 

## Training
To perform the training, run the following
```
python RGDM_train.py --splits splits/limit256.csv 
python DDPM_train.py --splits splits/limit256.csv 
```
These create a new log directory below `./workdir/`. The configuration and checkpoints during the training are saved there.
