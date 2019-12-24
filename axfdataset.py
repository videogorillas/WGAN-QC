import glob
import random

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

from axf import AXF


def _randomCrop(x, cropsize=128):
    dh = x.shape[0] - cropsize - 1
    dw = x.shape[1] - cropsize - 1
    rx = random.randint(0, dh)
    ry = random.randint(0, dw)
    return x[rx:rx + cropsize, ry:ry + cropsize, :]


def _toTensor(x):
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).float()
    return x


class AXFTextureDataset(Dataset):
    def __init__(self, path, image_size=128):
        self.image_size = image_size
        self.texturefiles = glob.glob(path + '*.axf')
        self.layerRanges = [(0, 1), (1, 4), (4, 7), (7, 10), (10, 11)]
        self.layerKeys = ['transparency_alpha', 'diffuse_color', 'diffuse_normal', 'specular_color', 'specular_lobes']

    def __len__(self):
        return 4200

    def __getitem__(self, index):
        with h5py.File(self.texturefiles[index % len(self.texturefiles)], 'r') as f:
            axf = AXF(f)
            ta = axf.transparency_alpha[0]
            dc = axf.diffuse_color[0]
            dn = axf.diffuse_normal[0]
            sc = axf.specular_color[0]
            sl = axf.specular_lobes[0]
            cat = np.concatenate([ta, dc, dn, sc, sl], axis=-1)
            crop = _randomCrop(cat, cropsize=self.image_size)
            crop = _toTensor(crop)
            if random.random() > 0.5:
                crop = crop.flip(2)
            return crop, crop


if __name__ == '__main__':
    d = AXFTextureDataset('/home/zhukov/clients/bru/textureGAN/data/AxF/')
    print(d[0].shape)
