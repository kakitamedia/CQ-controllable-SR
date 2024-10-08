import argparse
import os
import shutil
import numpy as np
import datetime
import wandb
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR

import utils
import models
import datasets
import losses

from utils import log
from test import do_test


def parse_args():
    parser = argparse.ArgumentParser(description='pytorch training code')
    parser.add_argument('--config', type=str, default='', metavar='FILE', help='path to config file')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--name', type=str, default=None, help='run name')
    parser.add_argument('--tag', type=str, default=None, help='tag for run')
    parser.add_argument('--eval_step', type=int, default=10, help='frequency of evaluation')
    parser.add_argument('--save_step', type=int, default=100, help='frequency of saving')
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids to use')
    parser.add_argument('--num_workers', type=int, default=16, help='the number of threads for data loader')

    return parser.parse_args()

def train():
    timer = utils.Timer()
    log('Loading Datasets...')
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    data_loader = make_data_loaders()
    log(f'Done. It took {utils.time_text(timer.t())}')

    log('Preparing Training...')
    timer.s()
    model, optimizer, scheduler, scaler, epoch_start = prepare_training()
    log(f'Done. It took {utils.time_text(timer.t())}')

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    epoch_max = config['epoch_max']
    max_val_v = -1e18

    # training loop
    for epoch in range(epoch_start, epoch_max+1):
        timer.s()
        log_info = [f'epoch {epoch}/{epoch_max}']

        # epoch
        loss, loss_dict = do_train(model, optimizer, scaler, data_loader)
        if scheduler is not None:
            scheduler.step()

        log_info.append(f'train: loss={loss:.4f}')

        if not args.debug:
            # save model
            if num_gpus > 1:
                model_ = model.module.model
            else:
                model_ = model.model
            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            save_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'epoch': epoch
            }
            torch.save(save_file, os.path.join(save_path, 'epoch-last.pth'))

            # save model every save_step
            if (args.save_step is not None) and (epoch % args.save_step == 0):
                torch.save(save_file,os.path.join(save_path, f'epoch-{epoch}.pth'))

        # validatation
        if (args.eval_step is not None) and (epoch % args.eval_step == 0):
            log('Validating...')
            model.eval()
            val_result, _ = do_test(model, data_loader['val'], validation=True)

            log_info.append(f'val: psnr={val_result:.4f}')

            # save best model
            if (val_result > max_val_v) and (not args.debug):
                max_val_v = val_result
                torch.save(save_file, os.path.join(save_path, 'epoch-best.pth'))

        # logging for wandb
        if not args.debug:
            log_dict = {'train/loss': loss, 'train/lr': optimizer.param_groups[0]['lr']}
            log_dict.update({f'train/{k}': v for k, v in loss_dict.items()})
            wandb.log(log_dict, step=epoch)
            if (args.eval_step is not None) and (epoch % args.eval_step == 0):
                log_dict = {'val/psnr': val_result}
                wandb.log(log_dict, step=epoch)

        # logging for log file
        log(', '.join(log_info))

def do_train(model, optimizer, scaler, data_loader):
    model.train()
    train_loss = utils.Avarager()
    train_loss_dict = {}

    for i, (inputs, targets) in enumerate(tqdm(data_loader['train'])):
        optimizer.zero_grad()
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}

        with torch.cuda.amp.autocast(enabled=config.get('amp'), dtype=torch.bfloat16):
            loss, loss_dict = model(inputs, targets)

        loss = loss.mean()
        scaler.scale(loss).backward()

        # gradient clipping
        if config.get('clip_grad') is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['clip_grad'])

        scaler.step(optimizer)
        scaler.update()
        # print(loss.item())
        train_loss.add(loss.item())

        for k, v in loss_dict.items():
            if train_loss_dict.get(k) is None:
                train_loss_dict[k] = utils.Avarager()
            train_loss_dict[k].add(v.item())

    return train_loss.item(), {k: v.item() for k, v in train_loss_dict.items()}

def make_data_loader(spec, tag=''):
    assert tag in ['train', 'val'], f'tag={tag} is not supported. Use \'train\' or \'val\'.'

    dataset = datasets.make(spec['dataset'])
    augmenter = datasets.augmentations.make(spec['augmenter']) if spec.get('augmenter') else None
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset, 'augment': augmenter})

    log(f'{tag} dataset size={len(dataset)}')
    for k, v in dataset[0][0].items():
        log(f'  {k}: shape={tuple(v.shape)}')

    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=args.num_workers, shuffle=(tag=='train'), pin_memory=True)
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return {'train': train_loader, 'val': val_loader}

def prepare_training():
    if resume_from is not None:
        log(f'Resume from {resume_from}')
        sv_file = torch.load(resume_from)
        model = models.make(sv_file['model'], load_sd=True)
        loss = losses.make(config['loss'], args={'model': model})
        model = ModelWithLoss(model, loss).cuda()
        optimizer = utils.make_optimizer(filter(lambda p:p.requires_grad, model.parameters()), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            scheduler = None
        else:
            scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            scheduler.step()

    else:
        model = models.make(config['model'])
        loss = losses.make(config['loss'], args={'model': model})
        model = ModelWithLoss(model, loss).cuda()
        optimizer = utils.make_optimizer(filter(lambda p:p.requires_grad, model.parameters()), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            scheduler = None
        else:
            scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    scaler = torch.cuda.amp.GradScaler(enabled=config.get('amp'))
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, scheduler, scaler, epoch_start


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(*args, **kwargs)
        else:
            return self.test_forward(*args, **kwargs)

    def train_forward(self, x, target):
        return self.loss_fn(self.model(**x), target)

    def test_forward(self, x):
        return self.model(**x)


def main():
    global args, config, save_path, resume_from
    args = parse_args()

    # set save path
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    resume_from = None
    remove = True

    if os.path.exists(save_path):
        if input(f'{save_path} exists, resume? (y/[n]): ') == 'y':
            log(f'config file is switched to {os.path.join(save_path, "config.yaml")}')
            args.config = os.path.join(save_path, 'config.yaml')
            resume_from = os.path.join(save_path, 'epoch-last.pth')
            remove = False
    utils.set_save_path(save_path, remove=remove)

    # load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set random seed
    seed = config.get('SEED')
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        if seed:
            torch.cuda.manual_seed(seed)
    else:
        raise Exception('GPU not found.')

    # set logging utils and wandb
    if not args.debug:
        if config.get('wandb_id') is None:
            config['wandb_id'] = wandb.util.generate_id()

        wandb.init(project='recurrent_lte', entity='kakita', id=config['wandb_id'], \
                   config=dict(yaml=config), name=save_name, resume='allow')
        # wandb.run.save()

        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    train()


if __name__ == '__main__':
    main()
