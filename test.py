import argparse
import os
import numpy as np
from tqdm import tqdm
import yaml
import math

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

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids to use')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--num_pred', type=int, default=None)
    parser.add_argument('--save_img', action='store_true', help='save images to the save_dir')
    parser.add_argument('--save_fourier', action='store_true', help='save fourier features to the save_dir')

    return parser.parse_args()

def test():
    timer = utils.Timer()
    print('Loading Datasets...')
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    dataset_spec = config['test_dataset']
    dataset = datasets.make(dataset_spec['dataset'])
    dataset = datasets.make(dataset_spec['wrapper'], args={'dataset': dataset})
    data_loader = DataLoader(dataset, batch_size=dataset_spec['batch_size'],\
        num_workers=args.num_workers, pin_memory=True)
    print(f'Done. It took {utils.time_text(timer.t())}')

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

    psnr, lpips, inference_time = do_test(model, data_loader, save_dir=save_path, batch_size=config.get('eval_bsize'))
    print(f'PSNR: {psnr}, LPIPS: {lpips}, Inference Time: {inference_time}')

    # output results to the txt file
    with open(os.path.join(save_path, 'results.txt'), 'w') as f:
        f.write(f'PSNR: {psnr}, LPIPS: {lpips}')

def do_test(model, data_loader, save_dir=None, batch_size=None, validation=False):
    model.eval()

    psnr = utils.Avarager()
    lpips = utils.Avarager()
    inference_time = utils.Avarager()
    timer = utils.Timer()
    if not validation:
        lpips_net = LPIPS(net='alex').cuda()

    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}

        timer.s()
        with torch.no_grad():
            if batch_size is None:
                preds = model(inputs)
            else:
                preds = batched_predict(model, inputs, batch_size)
        inference_time.add(timer.t())
        # # TODO: for debugging. remove it later
        # if config['test_dataset']['dataset']['args'].get('channel_flip'):
        #     preds['recon'] = torch.flip(preds['recon'], dims=[-1])
        #     targets['gt_img'] = torch.flip(targets['gt_img'], dims=[-1])
        #     targets['gt_rgb'] = torch.flip(targets['gt_rgb'], dims=[-1])

        if validation:
            preds['recon'] = utils.denormalize(preds['recon']).clamp_(0, 1)
            targets['gt_rgb'] = utils.denormalize(targets['gt_rgb'])
            psnr.add(utils.calc_psnr(preds['recon'], targets['gt_rgb']).item(), inputs['inp'].shape[0])

        else:
            preds['recon'] = preds['recon'].cuda()
            # reshape predictions
            ih, iw = inputs['inp'].shape[-2:]
            s = math.sqrt(inputs['coord'].shape[1] / (ih * iw))
            shape = [inputs['inp'].shape[0], round(ih * s), round(iw * s), -1]
            preds['recon'] = preds['recon'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            preds['recon'] = preds['recon'][..., :targets['gt_img'].shape[-2], :targets['gt_img'].shape[-1]]

            # discretize predictions
            preds['recon'] = utils.denormalize(preds['recon']).clamp_(0, 1)
            for j in range(preds['recon'].shape[0]):
                preds['recon'][j] = utils.discretize(preds['recon'][j])
            preds['recon'] = utils.normalize(preds['recon'])

            # calculate LPIPS
            lpips.add(lpips_net(preds['recon'], targets['gt_img']).mean().item(), inputs['inp'].shape[0])

            # calculate PSNR
            preds['recon'] = utils.denormalize(preds['recon']).clamp_(0, 1)
            targets['gt_img'] = utils.denormalize(targets['gt_img'])

            res = utils.calc_psnr(preds['recon'], targets['gt_img'])
            # print(res)
            psnr.add(res, inputs['inp'].shape[0])

            if (save_dir is not None) and args.save_img:
                for j in range(preds['recon'].shape[0]):
                    save_imgs(preds['recon'][j], os.path.join(save_dir, f'pred_{str(i).zfill(4)}_{res.item():.2f}.png'))
                    # needed = ['coef', 'freq', 'phase']
                    # save = {k: v for k, v in preds.items() if k in needed}
                    if args.save_fourier:
                        save = {k: v for k, v in preds.items()}
                        torch.save(save, os.path.join(save_dir, f'preds_{i}.pth'))

    return psnr.item(), lpips.item(), inference_time.item()

def batched_predict(model, inputs, batch_size):
    with torch.no_grad():
        inp, coord, cell = inputs['inp'], inputs['coord'], inputs['cell']
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = {}
        while ql < n:
            qr = min(ql + batch_size, n)
            pred = model.query(coord[:, ql:qr], cell[:, ql:qr])
            for k, v in pred.items():
                if k not in preds:
                    preds[k] = []
                if type(v) is list:
                    v = [x.cpu() for x in v]
                else:
                    v = v.cpu()
                preds[k].append(v)
            ql = qr

        for k, v in preds.items():
            if type(v[0]) is list:
                for i in range(len(v[0])):
                    preds[k] = [torch.cat([x[i] for x in v], dim=1)]
            else:
                preds[k] = torch.cat(v, dim=1)
    return preds

def save_imgs(img, path):
    img = torchvision.transforms.ToPILImage()(img.cpu())
    img.save(path)

def main():
    global args, config, save_path
    args = parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
