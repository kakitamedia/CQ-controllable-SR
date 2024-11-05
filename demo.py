import argparse
import os
import numpy as np
from tqdm import tqdm
import yaml
import math
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler
import torchvision

from lpips import LPIPS

import models
import losses
import datasets
import utils
from test import batched_predict, save_imgs

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids to use')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--num_pred', type=int, default=None)
    parser.add_argument('--scale_factors', type=str, default='4.0', help='comma-separated scale factors')
    parser.add_argument('--batch_size', type=int, default=300000)

    return parser.parse_args()

def test():
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
    if args.num_pred is not None:
        model.predictor.num_pred = args.num_pred
        model.num_pred = args.num_pred
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

        recon = preds['recon'].cuda()
        shape = [recon.shape[0], h, w, -1]
        recon = recon.view(*shape).permute(0, 3, 1, 2).contiguous()
        recon = utils.denormalize(recon).clamp_(0, 1)
        for j in range(recon.shape[0]):
            recon[j] = utils.discretize(recon[j])
        for j in range(recon.shape[0]):
            # save_imgs(recon[j], os.path.join(args.save_dir, f'pred_{str(i).zfill(4)}.png'))
            save_imgs(recon[j], os.path.join(args.save_dir, filename))


def main():
    global args, config, save_path
    args = parse_args()

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
    else:
        raise Exception('GPU not found.')

    # set save path
    save_name = args.save_dir
    assert len(save_name) > 0, 'save_dir must be specified.'
    # save_path = os.path.join('./save', save_name)
    save_path = save_name

    utils.ensure_path(save_path, remove=True)

    test()

if __name__ == '__main__':
    main()
