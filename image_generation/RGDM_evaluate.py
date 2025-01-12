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

from RGDM_diffusion import RGDiffusionSampler
from model import UNet
from dataset import get_dataset

# clean-fid
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.resize import build_resizer

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--num_images', type=int, required=True)
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


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def fidstats_ffhq64(fid_model, device, num_gen=50000, batch_size=128):
    # make dataloader
    dataset = get_dataset(dataset_key='ffhq64')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True)
    data_iter = iter(dataloader)
    # resizing function
    fn_resize = build_resizer(mode='clean')

    # calculate feature
    num_iters = int(np.ceil(num_gen / batch_size))
    feats = []
    for idx in tqdm(range(num_iters), desc="FID of FFHQ dataset"):
        flag = 0
        try:
            img_batch = next(data_iter).float()
            if img_batch.shape[0] < batch_size:
                logging.info("dataloder has been used up.")
                flag = 1
        except StopIteration:
            logging.info("dataloder has been used up.")
            flag = 1
        if flag != 0:
            break

        img_batch = (img_batch+1.)/2.
        img_batch = torch.clip(img_batch*255., 0, 255).to(torch.uint8)
        # resize image_batch to calculate clean FID
        img_batch_res = torch.zeros(batch_size, 3, 299, 299)
        for id in range(batch_size):
            curr_img = img_batch[id]
            img_np = curr_img.numpy().transpose((1, 2, 0))
            img_res = fn_resize(img_np)
            img_batch_res[id] = torch.tensor(img_res.transpose((2, 0, 1)))
        # get feature quantity
        feat = fid_model(img_batch_res.to(device)).detach().cpu().numpy()
        feats.append(feat)

    # compute mean and covariance
    np_feats = np.concatenate(feats)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    np.savez('FID_features/ffhq64_feat.npz', mu=mu, sigma=sigma)

    return mu, sigma


def fidstats_cifar10(split='train'):
    mu, sigma = get_reference_statistics(
        name="cifar10",
        res=32,
        mode="clean",
        split=split
    )
    return mu, sigma


def fidstats_gen_model(sampler, fid_model, num_images, device):
    with torch.no_grad():
        # resizing function
        fn_resize = build_resizer(mode='clean')

        batch_size = FLAGS.batch_size
        num_iters = int(np.ceil(num_images / batch_size))
        
        feats = []
        for _ in tqdm(range(num_iters), desc="model fid stats:"):
            # generate image
            x0 = sampler(batch_size, device)
            x0 = (x0 + 1.) / 2.
            x0 = torch.clip(x0*255., 0, 255).to(torch.uint8)
            # resize x0 to calculate clean FID
            x0_res = torch.zeros(batch_size, 3, 299, 299)
            for id in range(batch_size):
                curr_img = x0[id]
                img_np = curr_img.detach().cpu().numpy().transpose((1, 2, 0))
                img_res = fn_resize(img_np)
                x0_res[id] = torch.tensor(img_res.transpose((2, 0, 1)))
            # get feature quantity
            feat = fid_model(x0_res.to(device)).detach().cpu().numpy()
            feats.append(feat)
        
        # compute mean and covariance
        np_feats = np.concatenate(feats)
        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        return mu, sigma


def compute_fid(sampler, num_images, device, dataset_key):
    
    # load feature model of FID
    logging.info("Loading feature model of FID")
    fid_model = build_feature_extractor(mode="clean", device=device)
    
    # get dataset fid stats
    if dataset_key == 'ffhq64':
        path = 'FID_features/ffhq64_feat.npz'
        if os.path.exists(path):
            data = np.load(path)
            mu_data, sigma_data = data['mu'], data['sigma']
        else:
            mu_data, sigma_data = fidstats_ffhq64(fid_model, device)
    elif dataset_key == 'cifar10':
        mu_data, sigma_data = fidstats_cifar10(split='train')

    # get model fid stats
    logging.info("Computing fid stats of generative model")
    mu, sigma = fidstats_gen_model(sampler, fid_model, num_images, device=device)

    fid = frechet_distance(mu, sigma, mu_data, sigma_data)
    return fid


def eval(num_images=None):
    # ckpt_path
    ckpt_path = info_args.ckpt
    logging.info("FID caclulation\n\nRunning on {}".format(device))
    logging.info("ckpt_path = ".format(ckpt_path))
    
    # load model
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    model.load_state_dict(torch.load(ckpt_path)['ema_model'])  # ema, not net_model
    model.eval()
    
    # define sampler
    sampler = RGDiffusionSampler(
        model=model, T=FLAGS.T, L=FLAGS.img_size,
        beta_1=FLAGS.beta_1, beta_2=FLAGS.beta_T,
        m0=FLAGS.m0, mL=FLAGS.mL, k_cutoff=FLAGS.k_cutoff).to(device)

    # compute FID
    fid = compute_fid(sampler, num_images, device, FLAGS.dataset_key)

    # recored fid
    path = "rgdm_fid_test.txt"
    if os.path.exists(path):
        with open(path, mode="a") as f:
            f.write("{} {}\n".format(ckpt_path, fid))
    else:
        with open(path, mode="w") as f:
            f.write("ckpt  fid\n")
            f.write("{} {}\n".format(ckpt_path, fid))


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    eval(info_args.num_images)
