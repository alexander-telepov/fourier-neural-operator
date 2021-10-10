import numpy as np
from tqdm import tqdm
import os
from navier_stokes import navier_stokes_2d
import argparse


def generate_dataset(dataset, num_samples, path, params):
    if dataset == 'navier_stocks':
        generate_sample = navier_stokes_2d
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    os.mkdir(os.path.join(path, dataset))
    l = len(str(num_samples))

    for i in tqdm(range(num_samples)):
        input, solution = generate_sample(i, **params)
        np.save(os.path.join(path, dataset, f"input_{str(i).rjust(l, '0')}"), input.astype(np.float32))
        np.save(os.path.join(path, dataset, f"solution_{str(i).rjust(l, '0')}"), solution.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['navier_stocks'], default='navier_stocks')
    parser.add_argument('--num_samples', type=int, default=1200)
    parser.add_argument('--path', type=str, default='../datasets', help='path to folder with datasets')
    parser.add_argument('--s', type=int, default=256, help='spatial resolution')
    parser.add_argument('--T', type=float, default=50.0, help='end time')
    parser.add_argument('--t', type=int, default=50, help='time resolution')
    args = parser.parse_args()
    generate_dataset(args.dataset, args.num_samples, args.path, vars(args))


if __name__ == '__main__':
    main()
