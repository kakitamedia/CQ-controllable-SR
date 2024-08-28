import random
import math
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples, normalize

import warnings

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    def __init__(self, dataset, augment=None, inp_size=None, scale_min=1.0, scale_max=4.0, sample_q=None, patch_crop=None, pad_mode='zero'):
        self.dataset = dataset

        assert not (sample_q is not None and patch_crop is not None),\
            'sample_q and patch_crop cannot be used at the same time'

        self.inp_size = inp_size
        self.augment = augment
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.patch_crop = patch_crop
        self.pad_mode = pad_mode

        if inp_size is not None:
            self.max_size = inp_size * scale_max
        else:
            self.max_size = None

    def __getitem__(self, idx):
        hr_img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        # if inp_size is not specified, use the original size
        if self.inp_size is None:
            height_lr = math.floor(hr_img.shape[-2] / s + 1e-9)
            width_lr = math.floor(hr_img.shape[-1] / s + 1e-9)
            hr_img = hr_img[:, :round(height_lr * s), :round(width_lr * s)]
            lr_img = resize_fn(hr_img, (height_lr, width_lr))

            height_hr, width_hr = hr_img.shape[-2:]

        # if inp_size is specified, crop hr_img to the corresponding size
        else:
            height_lr = width_lr = self.inp_size
            height_hr = width_hr = round(width_lr * s)

            # if hr_img is smaller than the cropping size, pad hr_img
            if hr_img.shape[-2] < height_hr or hr_img.shape[-1] < width_hr:
                warnings.warn('The input image is smaller than the cropping size in the data loader. The images will be zero padded.')
                pad_height = max(0, width_hr - hr_img.shape[-2])
                pad_width = max(0, width_hr - hr_img.shape[-1])

                pad_top = random.randint(0, pad_height+1)
                pad_bottom = pad_height - pad_top
                pad_left = random.randint(0, pad_width+1)
                pad_right = pad_width - pad_left

                hr_img = F.pad(hr_img, (pad_left, pad_right, pad_top, pad_bottom), mode=self.pad_mode)

            crop_top = random.randint(0, hr_img.shape[-2] - width_hr)
            crop_left = random.randint(0, hr_img.shape[-1] - width_hr)
            hr_img = hr_img[:, crop_top:crop_top+width_hr, crop_left:crop_left+width_hr]

        # augmentation
        if self.augment is not None:
            # ugly code with strong dependency on albumentations
            hr_img = hr_img.permute(1, 2, 0).numpy()
            hr_img = self.augment(image=hr_img)['image']
            hr_img = torch.from_numpy(hr_img).permute(2, 0, 1)

        lr_img = resize_fn(hr_img, (height_lr, width_lr))
        hr_img, lr_img = normalize(hr_img), normalize(lr_img)
        hr_coord, hr_rgb = to_pixel_samples(hr_img)

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        if self.patch_crop is not None:
            crop_size = self.patch_crop
            crop_top = random.randint(0, height_hr - crop_size)
            crop_left = random.randint(0, width_hr - crop_size)
            crop_bottom = crop_top + crop_size
            crop_right = crop_left + crop_size

            sample_lst = torch.tensor([[i+crop_top+j*width_hr for i in range(crop_size)] for j in range(crop_size)]).view(-1)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / height_hr
        cell[:, 1] *= 2 / width_hr

        if self.max_size is not None:
            # pad the hr_img to the max_size for batch generation
            pad_height = max(0, self.max_size - hr_img.shape[-2])
            pad_width = max(0, self.max_size - hr_img.shape[-1])

            pad_top = random.randint(0, pad_height+1)
            pad_bottom = pad_height - pad_top
            pad_left = random.randint(0, pad_width+1)
            pad_right = pad_width - pad_left

            hr_img = F.pad(hr_img, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            # ugly code for generating padded_coord
            padded_coord = hr_coord.clone()
            padded_coord += 1.0
            padded_coord *= 0.5
            padded_coord *= torch.tensor([height_hr, width_hr]) / self.max_size
            padded_coord *= 2.0
            padded_coord -= 1.0

        inp = {
            'inp'   : lr_img,
            'coord' : hr_coord,
            'cell'  : cell,
        }
        target = {
            'gt_img' : hr_img,
            'gt_coord': padded_coord if self.max_size is not None else hr_coord,
            'gt_rgb': hr_rgb,
            'gt_cell': cell,
        }

        return inp, target

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    img = Image.open('load/div2k/DIV2K_train_HR/0001.png')

    img = transforms.ToTensor()(img)

    resized_img = resize_fn(img, (256, 256))
    print(img*255)
    print(resized_img*255)