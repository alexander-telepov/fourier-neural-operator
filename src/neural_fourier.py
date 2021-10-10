import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def compl_mul(ndim, a, b):
    if ndim == 1:
        op = partial(torch.einsum, "bix,iox->box")
    elif ndim == 2:
        op = partial(torch.einsum, "bctq,dctq->bdtq")
    elif ndim == 3:
        op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    else:
        raise ValueError(f'Unsupported ndim: {ndim}')

    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n1 = n_modes[0]

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, 2))

    def forward(self, x):
        n1 = self.n1
        batchsize = x.shape[0]

        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)

        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1) // 2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul(1, x_ft[:, :, :n1], self.weights1)

        out = torch.irfft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1), ))
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n1, self.n2 = n_modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, 2))

    def forward(self, x):
        n1, n2 = self.n1, self.n2
        batchsize = x.shape[0]

        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :n1, :n2] = compl_mul(2, x_ft[:, :, :n1, :n2], self.weights1)
        out_ft[:, :, -n1:, :n2] = compl_mul(2, x_ft[:, :, -n1:, :n2], self.weights2)

        out = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=(x.size(-2), x.size(-1)))
        return out


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n1, self.n2, self.n3 = n_modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, self.n3, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, self.n3, 2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, self.n3, 2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.n1, self.n2, self.n3, 2))

    def forward(self, x):
        n1, n2, n3 = self.n1, self.n2, self.n3
        batchsize = x.shape[0]

        x_ft = torch.rfft(x, 3, normalized=True, onesided=True)

        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :n1 , :n2 , :n3] = compl_mul(3, x_ft[:, :, :n1 , :n2 , :n3], self.weights1)
        out_ft[:, :, -n1:, :n2 , :n3] = compl_mul(3, x_ft[:, :, -n1:, :n2 , :n3], self.weights2)
        out_ft[:, :, :n1 , -n2:, :n3] = compl_mul(3, x_ft[:, :, :n1 , -n2:, :n3], self.weights3)
        out_ft[:, :, -n1:, -n2:, :n3] = compl_mul(3, x_ft[:, :, -n1:, -n2:, :n3], self.weights4)

        out = torch.irfft(out_ft, 3, normalized=True, onesided=True, signal_sizes=(x.size(-3), x.size(-2), x.size(-1)))
        return out


class SpatialBlock2d(nn.Module):
    def __init__(self, n_channels, activation=True, **kwargs):
        super(SpatialBlock2d, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, **kwargs)
        self.shortcut = nn.Conv1d(n_channels, n_channels, 1)
        self.bn = torch.nn.BatchNorm2d(n_channels)
        self.activation = activation
        self.n_channels = n_channels

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.n_channels, -1)).view(batchsize, self.n_channels, size_x, size_y)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out


class NeuralFourierBlock1d(nn.Module):
    def __init__(self, width, n_modes, activation=True):
        super(NeuralFourierBlock1d, self).__init__()
        self.conv = SpectralConv1d(width, width, n_modes)
        self.shortcut = nn.Conv1d(width, width, 1)
        self.bn = torch.nn.BatchNorm1d(width)
        self.activation = activation
        self.width = width

    def forward(self, x):
        batchsize = x.shape[0]
        size_x = x.shape[2]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out


class NeuralFourierBlock2d(nn.Module):
    def __init__(self, width, n_modes, activation=True):
        super(NeuralFourierBlock2d, self).__init__()
        self.conv = SpectralConv2d(width, width, n_modes)
        self.shortcut = nn.Conv1d(width, width, 1)
        self.bn = torch.nn.BatchNorm2d(width)
        self.activation = activation
        self.width = width

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out


class NeuralFourierBlock3d(nn.Module):
    def __init__(self, width, n_modes, activation=True):
        super(NeuralFourierBlock3d, self).__init__()
        self.conv = SpectralConv3d(width, width, n_modes)
        self.shortcut = nn.Conv1d(width, width, 1)
        self.bn = torch.nn.BatchNorm3d(width)
        self.activation = activation
        self.width = width

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out


class FourierNet1d(nn.Module):
    def __init__(self, n_layers, n_modes, width, t_in, t_out, pad=True, **kwargs):
        super(FourierNet1d, self).__init__()
        self.n_modes = n_modes
        self.width = width
        if pad:
            self.fc0 = nn.Linear(t_in + 1, width)
        else:
            self.fc0 = nn.Linear(t_in, width)

        layers = [NeuralFourierBlock1d(width, n_modes) for i in range(n_layers - 1)]
        layers.append(NeuralFourierBlock1d(width, n_modes, activation=False))
        self.backbone = nn.Sequential(*layers)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, t_out)

    def forward(self, x):
        out = self.fc0(x)

        out = out.permute(0, 2, 1)
        out = self.backbone(out)
        out = out.permute(0, 2, 1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


class SpatialNet2d(nn.Module):
    def __init__(self, n_layers, n_modes, width, t_in, t_out, pad=True, **kwargs):
        super(SpatialNet2d, self).__init__()
        self.n_modes = n_modes
        self.width = width
        if pad:
            self.fc0 = nn.Linear(t_in + 2, width)
        else:
            self.fc0 = nn.Linear(t_in, width)

        layers = [SpatialBlock2d(width, **kwargs) for i in range(n_layers - 1)]
        layers.append(SpatialBlock2d(width, activation=False, **kwargs))
        self.backbone = nn.Sequential(*layers)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, t_out)

    def forward(self, x):
        out = self.fc0(x)

        out = out.permute(0, 3, 1, 2)
        out = self.backbone(out)
        out = out.permute(0, 2, 3, 1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


class FourierNet2d(nn.Module):
    def __init__(self, n_layers, n_modes, width, t_in, t_out, pad=True, **kwargs):
        super(FourierNet2d, self).__init__()
        self.n_modes = n_modes
        self.width = width
        if pad:
            self.fc0 = nn.Linear(t_in + 2, width)
        else:
            self.fc0 = nn.Linear(t_in, width)

        layers = [NeuralFourierBlock2d(width, n_modes) for i in range(n_layers - 1)]
        layers.append(NeuralFourierBlock2d(width, n_modes, activation=False))
        self.backbone = nn.Sequential(*layers)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, t_out)

    def forward(self, x):
        out = self.fc0(x)

        out = out.permute(0, 3, 1, 2)
        out = self.backbone(out)
        out = out.permute(0, 2, 3, 1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


class FourierNet3d(nn.Module):
    def __init__(self, n_layers, n_modes, width, t_in, t_out=None, pad=True, **kwargs):
        super(FourierNet3d, self).__init__()
        self.n_modes = n_modes
        self.width = width
        if pad:
            self.fc0 = nn.Linear(t_in + 3, width)
        else:
            self.fc0 = nn.Linear(t_in, width)

        layers = [NeuralFourierBlock3d(width, n_modes) for i in range(n_layers - 1)]
        layers.append(NeuralFourierBlock3d(width, n_modes, activation=False))
        self.backbone = nn.Sequential(*layers)
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.fc0(x)

        out = out.permute(0, 4, 1, 2, 3)
        out = self.backbone(out)
        out = out.permute(0, 2, 3, 4, 1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out.squeeze()
