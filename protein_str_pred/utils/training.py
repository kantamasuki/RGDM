import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from scipy.ndimage import gaussian_filter1d

from .logging import get_logger

logger = get_logger(__name__)


def ema(source, target, decay=0.9999):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def get_scheduler(args, optimizer):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                    start_factor=args.lr_start, end_factor=1.0, total_iters=args.warmup_dur)
    constant = torch.optim.lr_scheduler.ConstantLR(optimizer,
                    factor=1., total_iters=args.constant_dur)
    decay = torch.optim.lr_scheduler.LinearLR(optimizer,
                    start_factor=1., end_factor=args.lr_end, total_iters=args.decay_dur)
    return torch.optim.lr_scheduler.SequentialLR(optimizer,
                    schedulers=[warmup, constant, decay], milestones=[args.warmup_dur, args.warmup_dur+args.constant_dur])


def epoch(args, model, ema_model, loader, optimizer=None, scheduler=None, device='cpu', print_freq=1000):
    if optimizer is not None: model.train()
    else: model.eval()
    
    loader = tqdm.tqdm(loader, total=len(loader))
    log = {'step': [], 'loss': []}
    
    # This block performs training for one epoch.
    # The function 'iter_' updates model parameters
    for i, data in enumerate(loader):
        data = data.to(device)
        try:
            if data.skip:
                logger.warning(f"Skipping batch")
                continue
            data, loss = iter_(model, ema_model, data, optimizer, scheduler)
            
            with torch.no_grad():
                log['step'].append(float(data.step.cpu().numpy()))
                log['loss'].append(float(loss.cpu().numpy()))

        except RuntimeError as e:
            if 'out of memory' in str(e):
                path = [d.path for d in data] if type(data) is list else data.path
                logger.warning(f'CUDA OOM, skipping batch {path}')
                for p in model.parameters():
                    if p.grad is not None: del p.grad  
                torch.cuda.empty_cache()
                continue
            
            else:
                logger.error("Uncaught error: " + str(e))
                
        if (i+1) % print_freq == 0:
            logger.info(f"Last {print_freq} iters: loss {np.mean(log['loss'][-print_freq:])}")
        
    return log


def iter_(model, ema_model, data, optimizer, scheduler):

    if optimizer is not None:

        model.zero_grad()

        # In 'model(data)', 'data.pred' is created.
        # 'data.pred' is the prediction of the colored/white noise.
        # Thus, we do not use 'pred' below.
        pred = model(data)
        loss = ((data.score - data.pred)**2).mean()
        loss.backward()
        
        ema(model, ema_model)

        # update model parameters
        if not np.isfinite(loss.item()):
            logger.warning(f"Nonfinite loss {loss.item()}; skipping")  
        elif loss.item() > 2000.0:
            logger.warning(f"Large loss {loss.item()}; skipping")
        else:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            except:
                logger.warning("Nonfinite grad, skipping")

    else: 
        with torch.no_grad():
            pred = model(data)
            loss = ((data.score - data.pred)**2).mean()

    return data, loss


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    return optimizer


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def save_loss_plot(log, path):
    x = np.array(log['step'])
    y1 = np.array(log['loss'])
    # y2 = np.array(log['base_loss'])
    order = np.argsort(x)
    # x, y1, y2 = x[order], y1[order], y2[order]
    x, y1 = x[order], y1[order]
    plt.scatter(x, y1, c='blue')
    # plt.scatter(x, y2, c='orange')
    plt.plot(x, gaussian_filter1d(y1, 1), c='blue')
    # plt.plot(x, gaussian_filter1d(y2, 1), c='orange')
    plt.ylim(bottom=0)
    plt.savefig(path)
    plt.clf()


 