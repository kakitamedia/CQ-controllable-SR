import os
import json
import pickle
from PIL import Image
import imageio
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
import utils

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, root_path, cache='none', split_file=None, split_key=None, first_k=None, repeat=1):
        self.cache = cache
        self.repeat = repeat

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            # if the cache is 'none', append the filenames to the list
            # slow and heavy disk utilization, but memory efficient
            if cache =='none':
                self.files.append(file)

            # if the cache is 'bin', save the images as a pickle files and append the filenames to the list,
            # faster than cache='none', but requires disk space and hevier disk utilization. same memory usage as cache='none'
            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path), '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            # if the cache is 'in_memory', append the loaded images to the list
            # fast but memory consuming
            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(Image.open(file).convert('RGB')))

            else:
                raise NotImplementedError(f'cache type {cache} is not supported')

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        # if the cache is 'in_memory', get the image directly from the list
        if self.cache == 'in_memory':
            x = x

        # if the cache is 'bin', load the image from the pickle file
        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255

        # if the cache is 'none', load the image from the file
        elif self.cache == 'none':
            x = transforms.ToTensor()(Image.open(x).convert('RGB'))

        return x

    def __len__(self):
        return len(self.files) * self.repeat


@register('image-folder-gray')
class ImageFolderGray(Dataset):
    def __init__(self, root_path, cache='none', split_file=None, split_key=None, first_k=None, repeat=1, channel_flip=False):
        self.cache = cache
        self.repeat = repeat
        self.channel_flip = channel_flip

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            # if the cache is 'none', append the filenames to the list
            # slow and heavy disk utilization, but memory efficient
            if cache =='none':
                self.files.append(file)

            # if the cache is 'bin', save the images as a pickle files and append the filenames to the list,
            # faster than cache='none', but requires disk space and hevier disk utilization. same memory usage as cache='none'
            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path), '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            # if the cache is 'in_memory', append the loaded images to the list
            # fast but memory consuming
            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(Image.open(file).convert('L')))

            else:
                raise NotImplementedError(f'cache type {cache} is not supported')

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        # if the cache is 'in_memory', get the image directly from the list
        if self.cache == 'in_memory':
            x = x

        # if the cache is 'bin', load the image from the pickle file
        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255

        # if the cache is 'none', load the image from the file
        elif self.cache == 'none':
            x = transforms.ToTensor()(Image.open(x).convert('L'))

        if self.channel_flip:
            x = x[[2, 1, 0]]

        return x

    def __len__(self):
        return len(self.files) * self.repeat


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]