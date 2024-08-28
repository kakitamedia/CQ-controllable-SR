import argparse
import yaml
import os
import math
from PIL import Image
import numpy as np
import einops

import torch
from torch.utils.data import DataLoader

from lpips import LPIPS

import utils
import models
import datasets

from test import do_test, batched_predict
from losses.partial_l1 import partial_reconstruction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--mode', default='descending')
    parser.add_argument('--num_basis', type=int, default=64)
    parser.add_argument('--gpu', default='0')

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
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_path = args.save_dir
    utils.ensure_path(save_path, remove=True)

    print('Loading Datasets...')
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    dataset_spec = config['test_dataset']
    dataset = datasets.make(dataset_spec['dataset'])
    dataset = datasets.make(dataset_spec['wrapper'], args={'dataset': dataset})
    data_loader = DataLoader(dataset, batch_size=dataset_spec['batch_size'], num_workers=8, pin_memory=True)
    print('Done.')

    print('Preparing Model...')
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    print('Done.')

    lpips_net = LPIPS(net='alex').cuda()
    avg_lpips = utils.Avarager()
    avg_psnr = utils.Avarager()

    for i, (inputs, targets) in enumerate(data_loader):
        print(f'Processing {i+1}/{len(data_loader)}...')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}

        with torch.no_grad():
            if config.get('eval_bsize') is None:
                preds = model(inputs)
            else:
                preds = batched_predict(model, inputs, config['eval_bsize'])
        lpips_per_sample = utils.Avarager()
        psnr_per_sample = utils.Avarager()
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

            gt_img = targets['gt_img']
            recon = recon.cuda()
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
            lpips = lpips_net(recon, gt_img).mean()
            lpips_per_sample.add(lpips.item(), inputs['inp'].shape[0])

            # calculate PSNR
            recon = utils.denormalize(recon)
            gt_img = utils.denormalize(gt_img)
            psnr = utils.calc_psnr(recon, gt_img).mean()
            psnr_per_sample.add(psnr.item(), inputs['inp'].shape[0])

            # save images
            if args.mode == 'descending':
                save_dir = os.path.join(save_path)
                os.makedirs(save_dir, exist_ok=True)
                for k in range(recon.shape[0]):
                    Image.fromarray((recon[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))\
                        .save(os.path.join(save_dir, f'{i * recon.shape[0] + k}.png'))
            elif args.mode == 'random':
                if loop_count < 3:
                    save_dir = os.path.join(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    for k in range(recon.shape[0]):
                        Image.fromarray((recon[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))\
                            .save(os.path.join(save_dir, f'{i * recon.shape[0] + k}_{loop_count}.png'))

        avg_lpips.add(lpips_per_sample.item())
        avg_psnr.add(psnr_per_sample.item())

    print(avg_psnr.item(), avg_lpips.item())
    with open(os.path.join(save_path, 'results.txt'), 'w') as f:
        f.write(f'PSNR: {avg_psnr.item()}, LPIPS: {avg_lpips.item()}')

if __name__ == '__main__':
    main()