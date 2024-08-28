import sys; sys.path.append('..')
import argparse
import os
import math
from functools import partial
import numpy as np
from PIL import Image

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

import datasets
import models
import utils
from losses.partial_l1 import partial_reconstruction

from test import do_test, batched_predict
from lpips import LPIPS
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--save_dir')
    parser.add_argument('--name', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=128)
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    global args, config, save_path

    args = parse_args()

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert num_gpus == 1, 'Multi-GPU is not supported.'

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
    else:
        raise Exception('GPU not found.')

    # load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set save path
    save_path = args.save_dir
    # save_name = args.save_dir
    # assert len(save_name) > 0, 'save_dir must be specified.'
    # save_path = os.path.join('./save', save_name)

    utils.ensure_path(save_path, remove=True)

    if not args.debug:
        wandb.init(project='recurrent_lte', entity='kakita', config=dict(yaml=config))
        wandb.run.name = f'num_samples/{args.name}'
        wandb.run.save()

    timer = utils.Timer()
    print('Loading Datasets...')
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    dataset_spec = config['test_dataset']
    dataset = datasets.make(dataset_spec['dataset'])
    dataset = datasets.make(dataset_spec['wrapper'], args={'dataset': dataset})
    data_loader = DataLoader(dataset, batch_size=dataset_spec['batch_size'],\
        num_workers=8, pin_memory=True)
    print(f'Done. It took {utils.time_text(timer.t())}')

    timer.s()
    print('Preparing Model...')
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    print(f'Done. It took {utils.time_text(timer.t())}')

    lpips_net = LPIPS(net='alex').cuda()
    start, end = args.start, args.end

    model.predictor.num_pred = end

    lpips_per_sample = [utils.Avarager() for _ in range(args.end+1)]
    psnr_per_sample = [utils.Avarager() for _ in range(args.end+1)]
    for i, (inputs, targets) in enumerate(data_loader):
        print(f'Processing {i+1}/{len(data_loader)}')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.inference_mode():
            if config.get('eval_bsize') is None:
                preds = model(inputs)
            else:
                preds = batched_predict(model, inputs, config['eval_bsize'])

        needed = ['coef', 'freq', 'phase']
        save = {k: v for k, v in preds.items() if k in needed}
        torch.save(save, os.path.join(save_path, f'preds_{i}.pth'))

        for j in tqdm(range(start, end+1)):
            recon = partial_reconstruction(preds, model, 'specify', end, length=j)
            gt_img = targets['gt_img']
            # reshape predictions
            ih, iw = inputs['inp'].shape[-2:]
            s = math.sqrt(inputs['coord'].shape[1] / (ih * iw))
            shape = [inputs['inp'].shape[0], round(ih * s), round(iw * s), -1]
            recon = recon.view(*shape).permute(0, 3, 1, 2).contiguous()
            recon = recon[..., :gt_img.shape[-2], :gt_img.shape[-1]]

            # discritize predictions
            recon = utils.denormalize(recon).clamp_(0, 1)
            for k in range(recon.shape[0]):
                recon[k] = utils.discretize(recon[k])
            recon = utils.normalize(recon)

            # calculate LPIPS
            recon = recon.cuda()
            lpips = lpips_net(recon, gt_img).mean()
            lpips_per_sample[j].add(lpips.item(), inputs['inp'].shape[0])

            # calculate PSNR
            recon = utils.denormalize(recon)
            gt_img = utils.denormalize(gt_img)
            psnr = utils.calc_psnr(recon, gt_img).mean()
            psnr_per_sample[j].add(psnr.item(), inputs['inp'].shape[0])

            # save images
            if args.save_img:
                save_dir = os.path.join(save_path, f'num_pred_{j}')
                os.makedirs(save_dir, exist_ok=True)
                for k in range(recon.shape[0]):
                    Image.fromarray((recon[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))\
                        .save(os.path.join(save_dir, f'{i * recon.shape[0] + k}.png'))

    if not args.debug:
        for i in range(start, end+1):
            wandb.log({'num_pred': i, 'psnr': psnr_per_sample[i].item(), 'lpips': lpips_per_sample[i].item()})

    # start, end = args.start, args.end

    # for i in range(start, end+1):
    #     model.predictor.num_pred = i
    #     model.num_pred = i
    #     save_dir = os.path.join(save_path, f'num_pred_{i}')
    #     os.makedirs(save_dir, exist_ok=True)
    #     psnr, lpips = do_test(model, data_loader, save_dir=save_dir, batch_size=config.get('eval_bsize'))

    #     if not args.debug:
    #         wandb.log({'num_pred': i, 'psnr': psnr, 'lpips': lpips})


if __name__ == '__main__':
    main()