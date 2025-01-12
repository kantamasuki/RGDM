import os
import warnings
import torch
import numpy as np
import pandas as pd
import copy

from model import get_model
from utils.RGDM_parsing import parse_train_args
from utils.dataset import RGDM_get_loader
from utils.logging import get_logger
from utils.training import (epoch, get_optimizer, get_scheduler,
                            save_loss_plot, save_yaml_file)

args = parse_train_args()
logger = get_logger(__name__)


def main():
    # set torch.device
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    # make log directory (=model_dir)
    if not args.dry_run:
        model_dir = os.path.join(args.workdir, str(args.time))
    else:
        model_dir = os.path.join(args.workdir, 'dry_run')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # save the training configuration as "./model_dir/args.yaml"
    yaml_file_name = os.path.join(model_dir, 'args.yaml')
    logger.info(f"Saving training args to {yaml_file_name}")
    save_yaml_file(yaml_file_name, args.__dict__)
    
    # load the splits (protein information)
    logger.info(f'Loading splits {args.splits}')
    splits = pd.read_csv(args.splits)
    try: splits = splits.set_index('path')   
    except: splits = splits.set_index('name')  

    # construct the model
    logger.info("Constructing model")
    model = get_model(args)
    enn_numel = sum([p.numel() for p in model.enn.parameters()])
    logger.info(f"ENN has {enn_numel} params")
    ema_model = copy.deepcopy(model)
    model = model.to(device)
    ema_model = ema_model.to(device)
    
    # define optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
  
    # make dataloader of training and validation datasets
    train_loader = RGDM_get_loader(args, splits, mode='train', shuffle=True)
    val_loader = RGDM_get_loader(args, splits, mode='val', shuffle=False)

    # run training
    run_training(args, model, ema_model, optimizer, scheduler,
                 train_loader, val_loader, device, model_dir=model_dir)


def run_training(args, model, ema_model, optimizer, scheduler,
                train_loader, val_loader, device, model_dir=None, 
                ep=1, best_val_loss = np.inf, best_epoch = 1):
    
    while ep <= args.epochs:
        # The function "epoch(...)" is the training function

        # training epoch
        logger.info(f"Starting training epoch {ep}")
        log = epoch(args, model, ema_model, train_loader,
                    optimizer=optimizer, scheduler=scheduler,
                    device=device, print_freq=args.print_freq)
        # training log
        train_loss = np.nanmean(log['loss'])
        logger.info(
            f"Train epoch {ep}: len {len(log['loss'])} loss {train_loss}")

        # validation epoch
        logger.info(f"Starting validation epoch {ep}")
        log = epoch(args, model, ema_model, val_loader,
                    device=device, print_freq=args.print_freq)
        # validation log
        val_loss = np.nanmean(log['loss'])
        logger.info(
            f"Val epoch {ep}: len {len(log['loss'])} loss {val_loss}")

        ### Save val loss plot
        png_path = os.path.join(model_dir, str(ep) + '.png')
        save_loss_plot(log, png_path)
        csv_path = os.path.join(model_dir, str(ep) + '.val.csv')
        pd.DataFrame(log).to_csv(csv_path)
        logger.info(f"Saved loss plot {png_path} and csv {csv_path}")

        ### Check if best epoch
        new_best = False
        if val_loss <= best_val_loss:
            best_val_loss = val_loss; best_epoch = ep
            logger.info(f"New best val epoch")
            new_best = True

        ### Save checkpoints
        state = {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epoch': ep,
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
        }
        if new_best:
            path = os.path.join(model_dir, 'best_model.pt')
            logger.info(f"Saving best checkpoint {path}")
            torch.save(state, path)

        if ep % args.save_freq == 0:
            path = os.path.join(model_dir, f'epoch_{ep}.pt')
            logger.info(f"Saving epoch checkpoint {path}")
            torch.save(state, path)

        path = os.path.join(model_dir, 'last_model.pt')
        logger.info(f"Saving last checkpoint {path}")
        torch.save(state, path)
            
        ep += 1
        
    logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
