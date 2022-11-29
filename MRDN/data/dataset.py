import torch.utils.data as data
import torch
import numpy as np
import os
import scipy.misc as misc


class DatasetFromFolder(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromFolder, self).__init__()
        self.In_paths = get_image_path(file_path['In'])
        self.GT_paths = get_image_path(file_path['GT'])

        assert(len(self.In_paths) == len(self.GT_paths))

    def __getitem__(self, item):
        In = np.load(self.In_paths[item])
        GT = np.load(self.GT_paths[item])
        In = np.transpose(In, (2,0,1))
        GT = np.transpose(GT, (2,0,1))

        return torch.from_numpy(In).float() / 255., torch.from_numpy(GT).float() / 255.

    def __len__(self):
        return len(self.In_paths)


def get_image_path(path):
    assert(os.path.isdir(path))
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            binary_path = os.path.join(dirpath, fname)
            files.append(binary_path)
    return files


def read_img(path):
    img = misc.imread(path)
    return img
