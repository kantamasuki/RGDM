import argparse
import os
import yaml
import torch
import pandas as pd

from utils.logging import get_logger
from model import DDPM_get_model
from utils.dataset import DDPM_get_loader
from utils.DDPM_sampling import DDPM_sampling_epoch

# Parameters for sampling
# When splits=limit256, split_key is either 'train' or 'val'. 
# When splits=cameo2022, any split_key is ok.
# When splits=cameo2022, embeddings_key must be changed to 'name'
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True) # directory where the chekpoint is saved
parser.add_argument('--ckpt', type=str, required=True) # chekpoint
parser.add_argument('--split_key', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--splits', type=str, default='splits/limit256.csv')
parser.add_argument('--embeddings_key', type=str, default='reference')
inf_args = parser.parse_args()

# Load configuration file used in the training
with open(f'{inf_args.model_dir}/args.yaml') as f:
    args = argparse.Namespace(**yaml.full_load(f))

# Save inf_args parameters to args
args.inference_mode = True
args.num_workers = inf_args.num_workers
args.num_samples = inf_args.num_samples
args.splits = inf_args.splits
args.embeddings_key = inf_args.embeddings_key

# Prepare logger and torch.device
logger = get_logger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():

    # Modify splits as needed
    logger.info(f'Loading splits {args.splits}')
    try: splits = pd.read_csv(args.splits).set_index('path')
    except: splits = pd.read_csv(args.splits).set_index('name')
    
    # Construct model
    logger.info("Constructing model")
    model = DDPM_get_model(args).to(device)
    ckpt = os.path.join(inf_args.model_dir, inf_args.ckpt)

    # Load weights from checkpoint (ema)
    logger.info(f'Loading weights from {ckpt}')
    state_dict = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['ema_model'], strict=True)
    ep = state_dict['epoch']

    # Make dataloader
    val_loader = DDPM_get_loader(args, splits, mode=inf_args.split_key, shuffle=False)
    
    # Prepare path to save the score of each sample 
    inf_name = f"{args.splits.split('/')[-1]}.ep{ep}.num{args.num_samples}."\
        + inf_args.split_key
    csv_path = os.path.join(inf_args.model_dir, f'{inf_name}/result.csv')
    
    # Run sampling
    _ = DDPM_sampling_epoch(
        args, model, val_loader.dataset, device=device, pdbs=True, elbo=False,
        model_dir=inf_args.model_dir, inf_name=inf_name, csv_path=csv_path)

        
if __name__ == '__main__':
    main()
