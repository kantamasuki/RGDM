# Protein structure prediction with the RGDM

This is the implementation of the RGDM and DDPM in protein structure prediction in [Generative Diffusion Model with Inverse Renormalization Group Flows](https://arxiv.org/abs/). Part of the codes is based on a previous work (['EigenFold' by B.Jing](https://github.com/bjing2016/EigenFold)).


## Version Information
We use Anaconda virtual environment. The environment and the versions of the GPU and Python libraries used in the work are as follows, while any reasonably recent version will be fine: 

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


## Installation
(This installation part is based on [github.com/bjing2016/EigenFold](https://github.com/bjing2016/EigenFold).) 
To run our codes, first install Pytorch and necessary libraries as
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-dct
pip install e3nn biopython
```

Also, one needs to download protein structures from Protein Data Bank ([PDB](https://www.rcsb.org)) by
```
bash download_pdb.sh ./data
python unpack_pdb.py --num_workers [N]
python make_splits.py
```
In addition to the downloaded PDB data, this will make `splits/limit256.csv`, which we also provide [here](https://drive.google.com/drive/folders/1S34ApICIXh7CAt5hcCV7UlXxIFSBD7w2?usp=drive_link).

Additionally, to create amino-sequence features, which are used in the training and generation steps of the diffusion models, one needs to install [OmegaFold](https://github.com/bjing2016/OmegaFold) by
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

## Paper results
All the trained models (i.e., checkpoints), the configuration of the training, and the sampled structures used in the paper are provided [here](https://drive.google.com/drive/folders/1S34ApICIXh7CAt5hcCV7UlXxIFSBD7w2?usp=drive_link). The data regarding the RGDM and DDPM are saved in `protein_paper_results/results/RGDM_trained` and `protein_paper_results/results/DDPM_trained`, respectively. To reproduce the results in the paper, download `paper_results/protien_paper_resutls/results` (i.e., make directory `./results`). 

## Sampling
To perform the sampling, run the following:
```
python RGDM_sample.py --model_dir results/RGDM_trained --ckpt epoch_9.pt --split_key ANY_KEY_IS_OK --splits splits/cameo2022.csv --embeddings_key name
python DDPM_sample.py --model_dir results/DDPM_trained --ckpt epoch_4.pt --split_key ANY_KEY_IS_OK --splits splits/cameo2022.csv --embeddings_key name
```
Sampled structures will be saved below `./results/RGDM_trained/` and `./results/DDPM_trained/`, respectively. 

## Training
To perform the training, run the following
```
python RGDM_train.py --splits splits/limit256.csv 
python DDPM_train.py --splits splits/limit256.csv 
```
These create a new log directory `./workdir/`, below which the checkpoints and the configuration during the training are saved. To change the hyperparameters during the training, modify `utlis/RGDM_parsing.py` and `utils/DDPM_parsing.py`, respectively.

## Sampling with AlphaFold2
To perform the sampling with AlphaFold2 in Fig. 3 in our paper, we used `./AlphaFold2_batch.ipynb` downloaded from [ColabFold](https://github.com/sokrypton/ColabFold). We verify that the program can be executed in Google Colab environment. The parameters we used are `msa_mode=MMseqs2 (UniRef+Environmental), num_models=1, num_recycles=1, stop_at_score=100, num_relax=0.`
