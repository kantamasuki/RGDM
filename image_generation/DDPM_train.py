import copy
import json
import os
import argparse
import warnings

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import trange
from ml_collections import ConfigDict

from DDPM_diffusion import DDPM_DiffusionSampler, DDPM_DiffusionTrainer
from dataset import get_dataset
from model import UNet


# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
info_args = parser.parse_args()

# read config file
config_path = info_args.config
with open(config_path, 'r') as f:
    config_data = json.load(f)    
FLAGS = ConfigDict(config_data)

# device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop_ffhq(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


def infiniteloop_cifar(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train():
    # make dataset, dataloader, and datalooper
    dataset = get_dataset(FLAGS.dataset_key)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    if FLAGS.dataset_key == 'ffhq64':
        datalooper = infiniteloop_ffhq(dataloader)
    if FLAGS.dataset_key == 'cifar10':
        datalooper = infiniteloop_cifar(dataloader)

    # # make model, optimizer, scheduler, trainer, and sampler
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = DDPM_DiffusionTrainer(
        net_model, 
        FLAGS.beta_1, FLAGS.beta_T, 
        FLAGS.T).to(device)
    ema_sampler = DDPM_DiffusionSampler(
        ema_model,
        FLAGS.beta_1, FLAGS.beta_T, 
        FLAGS.T, FLAGS.img_size).to(device)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    # make directories and save config
    if not os.path.exists(os.path.join(FLAGS.logdir)):
        os.makedirs(os.path.join(FLAGS.logdir))
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    if not os.path.exists(os.path.join(FLAGS.logdir, 'ckpt')):
        os.makedirs(os.path.join(FLAGS.logdir, 'ckpt'))
    if not os.path.exists(os.path.join(FLAGS.logdir, 'tensorboard')):
        os.makedirs(os.path.join(FLAGS.logdir, 'tensorboard'))
    with open(FLAGS.logdir + "/saved_config.json", "w") as json_file:
        json.dump(config_data, json_file, indent=4)

    # preparation
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(os.path.join(FLAGS.logdir, 'tensorboard'))
    writer.add_image('real_sample', grid)
    writer.flush()

    # run training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(FLAGS.sample_size, device)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(
                    FLAGS.logdir, 'ckpt/ckpt_{}.pt'.format(step//FLAGS.save_step)))

    writer.close()


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    train()
