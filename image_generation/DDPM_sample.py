import json
import os
import warnings
import argparse
import torch
from torchvision.utils import make_grid, save_image
from ml_collections import ConfigDict
from DDPM_diffusion import DDPM_DiffusionSampler
from model import UNet

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
info_args = parser.parse_args()

# read logdir/saved_config.json
config_path = info_args.config
with open(config_path, 'r') as f:
    config_data = json.load(f)    
FLAGS = ConfigDict(config_data)

# device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def sample():
    # checkpoint path
    ckpt_path = info_args.ckpt
    
    # load model
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    model.load_state_dict(torch.load(ckpt_path)['ema_model'])  # ema, not net_model
    model.eval()
    
    # define sampler
    sampler = DDPM_DiffusionSampler(
        model, 
        FLAGS.beta_1, FLAGS.beta_T, 
        FLAGS.T, FLAGS.img_size).to(device)

    # run sampling
    batch_size, nrow = 16, 4
    with torch.no_grad():
        x0 = sampler(batch_size, device).detach().cpu()
    grid = (make_grid(x0, nrow, padding=1) +1.)/2.
    path = 'sample/DDPM_{}_T{}.png'.format(FLAGS.dataset_key, FLAGS.T)
    save_image(grid, path)


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    sample()
