import os
import shutil
import time
import numpy as np

import torch
from torch.optim import SGD, Adam, RAdam, AdamW
from torchvision.transforms import Normalize, ToTensor, ToPILImage

_log_path = None

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def set_save_path(save_path, remove=True):
    global _log_path
    _log_path = save_path
    ensure_path(save_path, remove=remove)

yes = False
def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_') or input(f'{path} exists, remove? (y/[n]): ') == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'radam': RAdam,
        'adamw': AdamW,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

class Avarager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    img = img.contiguous()
    coord = make_coord(img.shape[-2:])
    rgb = img.view(img.shape[0], -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

# def fourier_decoding(coef, freq, phase, coord):
#     freq = torch.stack(torch.split(freq, 2, dim=-1), dim=-1)
#     freq = torch.mul(freq, coord.unsqueeze(-1))
#     freq = torch.sum(freq, dim=-2)
#     freq += phase
#     freq = torch.cat((torch.cos(np.pi*freq), torch.sin(np.pi*freq)), dim=-1)

#     return torch.mul(coef, freq)

import einops
def fourier_decoding(coef, freq, phase, coord):
    freq = einops.rearrange(freq, '... (l xy) -> ... xy l', xy=2)
    freq = torch.mul(freq, coord.unsqueeze(-1))
    freq = einops.reduce(freq, '... xy l -> ... l', 'sum')
    freq += phase
    freq = torch.cat((torch.cos(np.pi*freq), torch.sin(np.pi*freq)), dim=-1)

    return torch.mul(coef, freq)

_mean, _std = [0., 0., 0.], [1., 1., 1.]

def set_normalizer(mean, std):
    global _mean, _std
    _mean, _std = mean, std

def normalize(x):
    if x.shape[-3] == len(_mean):
        return Normalize(_mean, _std)(x)
    else:
        return unreshape_pixels(\
            Normalize(_mean, _std)(\
                reshape_pixels(x)))

def denormalize(x):
    if x.shape[-3] == len(_mean):
        return Normalize([-m/s for m, s in zip(_mean, _std)], [1/s for s in _std])(x)
    else:
        return unreshape_pixels(\
            Normalize([-m/s for m, s in zip(_mean, _std)], [1/s for s in _std])(\
                reshape_pixels(x)))

def reshape_pixels(x):
    return x.permute(0, 2, 1).unsqueeze(-1)

def unreshape_pixels(x):
    return x.squeeze(-1).permute(0, 2, 1)

def discretize(x):
    assert (x.min() >= 0 and x.max() <= 1)
    return ToTensor()(ToPILImage()(x))