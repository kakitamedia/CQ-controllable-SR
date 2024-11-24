import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import os
import math
from PIL import Image
import numpy as np
import einops
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler
import torchvision

from lpips import LPIPS

import utils
import models
import datasets

from test import do_test, batched_predict, save_imgs
from losses.partial_l1 import partial_reconstruction

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids to use')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--num_basis', type=int, default=None)
    parser.add_argument('--scale_factors', type=str, default='4.0', help='comma-separated scale factors')
    parser.add_argument('--batch_size', type=int, default=300000)
    parser.add_argument('--save_fourier', action='store_true', help='save fourier features to the save_dir')
    parser.add_argument('--mode', type=str, default='descending', help='mode for sorting the coefficients')

    return parser.parse_args()


def main():
    global args, config, save_path

    args = parse_args()

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert num_gpus == 1, 'Multi-GPU running is not supported.'

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        raise Exception('GPU not found.')

    args = parse_args()

    save_path = args.save_dir
    utils.ensure_path(save_path, remove=True)

    timer = utils.Timer()
    data_norm = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }
    utils.set_normalizer(**data_norm)

    timer.s()
    print('Preparing Model...')
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    print(f'Done. It took {utils.time_text(timer.t())}')

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    model.eval()
    batch_size = args.batch_size if args.batch_size > 0 else None
    scale_factors = [float(x) for x in args.scale_factors.split(',')]
    if len(scale_factors) == 1:
        scale_factors = [scale_factors[0], scale_factors[0]]

    for i, filename in tqdm(enumerate(sorted(os.listdir(args.input_dir)))):
        image = torchvision.transforms.ToTensor()(Image.open(os.path.join(args.input_dir, filename)).convert('RGB')).cuda().unsqueeze(0)
        h, w = int(image.shape[-2] * scale_factors[0]), int(image.shape[-1] * scale_factors[1])
        coord = utils.make_coord((h, w)).to(image.device).unsqueeze(0)
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2 / h
        cell[:, :, 1] *= 2 / w

        image = utils.normalize(image)

        inputs = {
            'inp' : image,
            'coord' : coord,
            'cell' : cell
        }

        with torch.no_grad():
            if batch_size is None:
                preds = model(inputs)
            else:
                preds = batched_predict(model, inputs, batch_size)

        if args.mode == 'descending':
            loop = 1
        else:
            loop = 3
        for loop_count in range(loop):
            freq = einops.rearrange(preds['freq'][0][0], 'b (l c xy) -> b xy c l', c=3, xy=2)
            coef = einops.rearrange(preds['coef'][0][0], 'b (l c) -> b c l', c=3)
            # coef = einops.rearrange(preds['coef'][0][0], 'b (n c xy) -> b xy c n', c=3, xy=2)
            phase = einops.rearrange(preds['phase'][0][0], 'b (l c) -> b c l', c=3)

            l = freq.shape[-1]

            if args.mode == 'descending':
                mag_R = coef[:, 0, :l] ** 2 + coef[:, 0, l:] ** 2
                mag_G = coef[:, 1, :l] ** 2 + coef[:, 1, l:] ** 2
                mag_B = coef[:, 2, :l] ** 2 + coef[:, 2, l:] ** 2
                sort_idx_R = torch.argsort(mag_R, descending=True)
                sort_idx_G = torch.argsort(mag_G, descending=True)
                sort_idx_B = torch.argsort(mag_B, descending=True)
            elif args.mode == 'random':
                sort_idx_R = torch.randperm(l).expand(coef.shape[0], -1)
                sort_idx_G = torch.randperm(l).expand(coef.shape[0], -1)
                sort_idx_B = torch.randperm(l).expand(coef.shape[0], -1)

            coef[:, 0, :l] = torch.gather(coef[:, 0, :l], -1, sort_idx_R)
            coef[:, 0, l:] = torch.gather(coef[:, 0, l:], -1, sort_idx_R)
            coef[:, 1, :l] = torch.gather(coef[:, 1, :l], -1, sort_idx_G)
            coef[:, 1, l:] = torch.gather(coef[:, 1, l:], -1, sort_idx_G)
            coef[:, 2, :l] = torch.gather(coef[:, 2, :l], -1, sort_idx_B)
            coef[:, 2, l:] = torch.gather(coef[:, 2, l:], -1, sort_idx_B)

            freq[:, 0, 0] = torch.gather(freq[:, 0, 0], -1, sort_idx_R)
            freq[:, 1, 0] = torch.gather(freq[:, 1, 0], -1, sort_idx_R)
            freq[:, 0, 1] = torch.gather(freq[:, 0, 1], -1, sort_idx_G)
            freq[:, 1, 1] = torch.gather(freq[:, 1, 1], -1, sort_idx_G)
            freq[:, 0, 2] = torch.gather(freq[:, 0, 2], -1, sort_idx_B)
            freq[:, 1, 2] = torch.gather(freq[:, 1, 2], -1, sort_idx_B)

            phase[:, 0] = torch.gather(phase[:, 0], -1, sort_idx_R)
            phase[:, 1] = torch.gather(phase[:, 1], -1, sort_idx_G)
            phase[:, 2] = torch.gather(phase[:, 2], -1, sort_idx_B)

            preds['freq'][0][0] = einops.rearrange(freq, 'b xy c n -> b (n c xy)')
            preds['coef'][0][0] = einops.rearrange(coef, 'b c l -> b (l c)')
            preds['phase'][0][0] = einops.rearrange(phase, 'b c l -> b (l c)')
            recon = partial_reconstruction(preds, model, 'specify', l, length=args.num_basis)

        # breakpoint()
        # recon = preds['recon'].cuda()
        shape = [recon.shape[0], h, w, -1]
        recon = recon.view(*shape).permute(0, 3, 1, 2).contiguous()
        recon = utils.denormalize(recon).clamp_(0, 1)
        for j in range(recon.shape[0]):
            recon[j] = utils.discretize(recon[j])
        for j in range(recon.shape[0]):
            # save_imgs(recon[j], os.path.join(args.save_dir, f'pred_{str(i).zfill(4)}.png'))
            save_imgs(recon[j], os.path.join(args.save_dir, filename))
            if args.save_fourier:
                save = {k: v for k, v in preds.items()}
                torch.save(save, os.path.join(args.save_dir, f'{os.path.splitext(filename)[0]}.pth'))


if __name__ == '__main__':
    main()