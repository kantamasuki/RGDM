import json
import os
import warnings
import argparse
import logging
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from ml_collections import ConfigDict

from DDPM_diffusion import DDPM_DiffusionSampler
from model import UNet
from dataset import get_dataset

# clean-fid
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.resize import build_resizer

# evaluate function
from RGDM_evaluate import compute_fid

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--num_images', type=int, required=True)
info_args = parser.parse_args()

# raed logdir/saved_config.json
config_path = info_args.config
with open(config_path, 'r') as f:
    config_data = json.load(f)    
FLAGS = ConfigDict(config_data)

# device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def eval(num_images=None):
    # ckpt path
    ckpt_path = info_args.ckpt
    logging.info("FID caclulation\n\nRunning on {}".format(device))
    
    # load ema model
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    model.load_state_dict(torch.load(ckpt_path)['ema_model'])
    model.eval()

    # define sampler
    sampler = DDPM_DiffusionSampler(
        model, 
        FLAGS.beta_1, FLAGS.beta_T, 
        FLAGS.T, FLAGS.img_size).to(device)

    # compute fid
    fid = compute_fid(sampler, num_images, device, FLAGS.dataset_key)

    # recored fid
    path = "ddpm_fid_test.txt"
    if os.path.exists(path):
        with open(path, mode="a") as f:
            f.write("{} {}\n".format(ckpt_path, fid))
    else:
        with open(path, mode="w") as f:
            f.write("fid\n")
            f.write("{} {}\n".format(ckpt_path, fid))


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    eval(info_args.num_images)
