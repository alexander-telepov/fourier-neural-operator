import os
import argparse
from tqdm import tqdm
import numpy as np
import scipy.io
import h5py
import torch


def parse_command_line():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', dest='mode', type=bool, help='True: mat_to_npy, False: npy_to_mat', default=True)
    parser.add_argument('--mat', dest='mat', type=str, help='path to mat file')
    parser.add_argument('--save', dest='save', type=str, help='save npy to here')
    parser.add_argument('--path', dest='path', type=str, help='save npy to here', default='../datasets')

    return parser.parse_args()


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


args = parse_command_line()
mode = args.mode
dataset = args.save
path = args.path

if mode:
    mat = args.mat
    reader = MatReader(mat, to_torch=False)
    u = reader.read_field('u')
    a = reader.read_field('a')
    
    num_samples = u.shape[0]
    os.mkdir(os.path.join(path, dataset))
    l = len(str(num_samples))
    for i in tqdm(range(num_samples)):
        input = a[i, :, :]
        solution = u[i, :, :]
        np.save(os.path.join(path, dataset, f"input_{str(i).rjust(l, '0')}"), input.astype(np.float32))
        np.save(os.path.join(path, dataset, f"solution_{str(i).rjust(l, '0')}"), solution.astype(np.float32))

