import torch
from neural_fourier import FourierNet1d, FourierNet2d, FourierNet3d, SpatialNet2d
from unet import UNet
from train import Trainer
from data import Data
from utils import parse_args, dump_config, mkdirs
import sys
from contextlib import redirect_stdout
import os
import random
import numpy as np


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_args():
    default_args = {
        "dataset": "ns_V1e-3_N1200_T50_1_s4",
        "net_arch": "3d",
        "batch_size": 10,
        "n_epochs": 500,
        "weight_decay": 0.0001,
        "learning_rate": 0.0025,
        "scheduler_step": 100,
        "scheduler_gamma": 0.5,
        "n_layers": 4,
        "n_modes_x": 4,
        "n_modes_y": 4,
        "n_modes_t": 4,
        "width": 20,
        "predictive_mode": "multiple_step",
        "S": 64,
        "s": 1,
        "t": 1,
        "num_samples": 1200,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "shuffle": "true",
        "pad_coordinates": "true",
        "seed": 42,
        "device": "cuda",
        "experiments": "../experiments",
        "datasets": "../datasets",
        "make_deterministic": "false"
    }

    return default_args


def main():
    config = sys.argv[1]
    args = parse_args(config)
    args = {**get_default_args(), **args}
    command = args['command']

    mkdirs(command, args)
    dump_config(command, args)

    if args['net_arch'] == '1d':
        net_class = FourierNet1d
        n_modes = (args['n_modes_x'])
        kwargs = {}
    elif args['net_arch'] == '2d':
        net_class = FourierNet2d
        n_modes = (args['n_modes_x'], args['n_modes_y'])
        kwargs = {}
    elif args['net_arch'] == '3d':
        net_class = FourierNet3d
        n_modes = (args['n_modes_x'], args['n_modes_y'], args['n_modes_t'])
        kwargs = {}
    elif args['net_arch'] == '2d_spatial':
        net_class = SpatialNet2d
        n_modes = -1
        kwargs = {'kernel_size': args['kernel_size'], 'padding': args['padding']}
    elif args['net_arch'] == 'unet':
        net_class = UNet
        n_modes = -1
        kwargs = {'kernel_size': args['kernel_size'], 'padding': args['padding']}
    else:
        raise ValueError(f'Unknown net_arch: {args["net_arch"]}')

    if args['predictive_mode'] == 'one_step':
        t_out = args['step']
    else:
        t_out = args['t_out']

    if args['make_deterministic'] == 'true':
        fix_seed(args['seed'])

    net = net_class(args['n_layers'], n_modes, args['width'], args['t_in'], t_out,
                    args['pad_coordinates'] == "true", **kwargs).to(args['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['scheduler_step'],
                                                gamma=args['scheduler_gamma'])

    data = Data(args)
    train_loader, val_loader, test_loader = data.get_dataloaders()

    trainer = Trainer(args, net, optimizer, scheduler, train_loader, val_loader, test_loader)

    with open(os.path.join(args['experiments'], args['exp_name'], f'log_{command}.txt'), 'w') as f:
        with redirect_stdout(f):
            if command == 'train':
                trainer.train()
            elif command == 'train&test':
                trainer.train()
                loss = trainer.test(test_loader).item()
                print(f'Test loss: {loss / len(test_loader)}')
            elif command == 'train&test&predict':
                trainer.train()
                loss = trainer.test(test_loader).item()
                print(f'Test loss: {loss / len(test_loader)}')
                trainer.predict(test_loader)
            elif command == 'test':
                trainer.load_model()
                loss = trainer.test(test_loader).item()
                print(f'Test loss: {loss / len(test_loader)}')
            elif command == 'predict':
                trainer.load_model()
                trainer.predict(test_loader)
            else:
                raise ValueError(f'Unknown command: {command}')


if __name__ == '__main__':
    main()
