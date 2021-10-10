import torch
import torch.utils.data as torch_data
from torchvision import transforms
import numpy as np
import os


class ToTensor(object):
    def __call__(self, sample):
        return list(map(torch.from_numpy, sample))


class Downsample(object):
    def __init__(self, s, t):
        self.s = s
        self.t = t

    def __call__(self, sample):
        shape, ndim = sample.shape, sample.ndim
        s, t = self.s, self.t
        slices = [slice(0, shape[i], s) for i in range(ndim - 1)]
        slices.append(slice(0, shape[ndim-1], t))
        return sample[tuple(slices)]


class NumOutTimesteps(object):
    def __init__(self, t_out):
        self.t_out = t_out

    def __call__(self, sample):
        input, label = sample
        t_out = self.t_out
        return input, label[..., :t_out]


class OutTimestepsRepeat(object):
    def __init__(self, t_out):
        self.t_out = t_out

    def __call__(self, sample):
        T = self.t_out
        input, label = sample
        input = input.reshape(input.shape[0], input.shape[1], 1, input.shape[2]).repeat([1, 1, T, 1])
        return input, label


class PadCoordinates1d(object):
    def __init__(self, S):
        self.S = S
        self.padding = torch.linspace(0, 1, S, dtype=torch.float32).reshape(S, 1)

    def __call__(self, sample):
        input, label = sample
        input = torch.cat((self.padding, input), dim=-1)
        return input, label


class PadCoordinates2d(object):
    def __init__(self, S):
        self.S = S
        self.padding = torch.empty(S, S, 2, dtype=torch.float32)
        self.padding[:, :, 0] = torch.linspace(0, 1, S, dtype=torch.float32).reshape(S, 1)
        self.padding[:, :, 1] = torch.linspace(0, 1, S, dtype=torch.float32).reshape(1, S)

    def __call__(self, sample):
        input, label = sample
        input = torch.cat((self.padding, input), dim=-1)
        return input, label


class PadCoordinates3d(object):
    def __init__(self, S, t_out):
        self.S = S
        self.t_out = t_out
        self.padding = torch.empty(S, S, t_out, 3, dtype=torch.float32)
        self.padding[:, :, :, 0] = torch.linspace(0, 1, S,       dtype=torch.float32)    .reshape(S, 1, 1)
        self.padding[:, :, :, 1] = torch.linspace(0, 1, S,       dtype=torch.float32)    .reshape(1, S, 1)
        self.padding[:, :, :, 2] = torch.linspace(0, 1, t_out+1, dtype=torch.float32)[1:].reshape(1, 1, t_out)

    def __call__(self, sample):
        input, label = sample
        input = torch.cat((self.padding, input), dim=-1)
        return input, label


class ContiniousRandomCut(object):
    def __init__(self, t_in, t_out):
        self.t_in = t_in
        self.t_out = t_out

    def __call__(self, sample):
        t_in, t_out = self.t_in, self.t_out
        sample = torch.cat(sample, dim=-1)
        start = np.random.randint(t_in - 1)
        return sample[..., start:start+t_out], sample[..., start+1:start+t_out+1]


class PDEDataset(torch_data.Dataset):
    def __init__(self, path, ids, l, t_in, downsampler, transform=None):
        super(PDEDataset, self).__init__()
        self.path = path
        self.ids = ids
        self.transform = transform
        self.l = l
        self.t_in = t_in
        self.downsampler = downsampler

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        solution = np.load(os.path.join(self.path, f"solution_{str(self.ids[idx]).rjust(self.l, '0')}.npy"))
        solution = self.downsampler(solution)
        t_in = self.t_in
        input, label = solution[..., :t_in], solution[..., t_in:]
        if self.transform is not None:
            input, label = self.transform((input, label))

        return input.float(), label.float()


class Data(object):
    def __init__(self, args):
        self.unpack_args(args)
        self.path = os.path.join(args['datasets'], args['dataset'])

    def unpack_args(self, args):
        for key, value in args.items():
            setattr(self, key, value)

    def inspect_folder(self):
        npy_files_length = list(map(len, filter(lambda x: str.endswith(x, '.npy'), os.listdir(self.path))))
        num_files = len(npy_files_length) // 2
        npy_files_length = set(npy_files_length)
        assert len(npy_files_length) == 2
        l1, l2 = min(npy_files_length), max(npy_files_length)
        l1, l2 = l1 - len(''.join(['input_', '.npy'])), l2 - len(''.join(['solution_', '.npy']))
        assert l1 == l2
        l = l1

        if self.num_samples > num_files:
            print(f'Not enough samples in {self.path}, num_samples was decreased to: {num_files}')
            self.num_samples = num_files

        assert self.num_samples >= self.batch_size
        if self.num_samples % self.batch_size != 0:
            print(f'Number of samples non multiple to batch_size, skip last non-full batch')
            self.num_samples -= self.num_samples % self.batch_size

        return l

    def get_transforms(self):
        basic_transforms = [
            NumOutTimesteps(self.t_out),
            ToTensor()
        ]

        if self.net_arch == "3d":
            basic_transforms.append(OutTimestepsRepeat(self.t_out))

        if self.pad_coordinates == 'true':
            if self.net_arch == "1d":
                basic_transforms.append(PadCoordinates1d(self.S))
            elif self.net_arch == "2d" or self.net_arch == "2d_spatial":
                basic_transforms.append(PadCoordinates2d(self.S))
            elif self.net_arch == "3d":
                basic_transforms.append(PadCoordinates3d(self.S, self.t_out))

        if self.predictive_mode == 'unet_step':
            basic_transforms.append(ContiniousRandomCut(self.t_in, self.t_out))

        transforms_train = transforms.Compose(basic_transforms)
        transforms_val = transforms.Compose(basic_transforms)
        transforms_test = transforms.Compose(basic_transforms)

        return transforms_train, transforms_val, transforms_test

    def get_ids(self):
        np.random.seed(self.seed)
        if self.shuffle == 'true':
            permutation = np.random.permutation(self.num_samples)
        elif self.shuffle == 'false':
            permutation = np.arange(self.num_samples)
        test_len = int(self.num_samples * self.test_ratio)
        test_len = test_len - test_len % self.batch_size
        val_len = int((self.num_samples - test_len) * self.val_ratio)
        val_len = val_len - val_len % self.batch_size
        train_len = self.num_samples - test_len - val_len
        train_ids = permutation[:train_len]
        val_ids = permutation[train_len:train_len+val_len]
        test_ids = permutation[train_len+val_len:train_len+val_len+test_len]

        return train_ids, val_ids, test_ids

    def get_dataloaders(self):
        l = self.inspect_folder()
        train_ids, val_ids, test_ids = self.get_ids()
        train_dataloader, val_dataloader, test_dataloader = None, None, None
        transforms_train, transforms_val, transforms_test = self.get_transforms()
        downsampler = Downsample(self.s, self.t)

        if len(train_ids) > 0:
            train_dataset = PDEDataset(self.path, train_ids, l, self.t_in, downsampler, transforms_train)
            train_dataloader = torch_data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if len(val_ids) > 0:
            val_dataset = PDEDataset(self.path, val_ids, l, self.t_in, downsampler, transforms_val)
            val_dataloader = torch_data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        if len(test_ids) > 0:
            test_dataset = PDEDataset(self.path, test_ids, l, self.t_in, downsampler, transforms_test)
            test_dataloader = torch_data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
