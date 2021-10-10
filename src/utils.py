import json
import os


def parse_args(file):
    with open(file, 'r') as f:
        return json.load(f)


def mkdirs(command, args):
    exp_folder = os.path.join(args['experiments'], args['exp_name'])
    if command == 'train' or command == 'train&test':
        os.mkdir(exp_folder)
        os.mkdir(os.path.join(exp_folder, 'tensorboard'))
    elif command == 'predict':
        os.makedirs(os.path.join(exp_folder, 'predictions'), exist_ok=True)
        os.mkdir(os.path.join(exp_folder, 'predictions', args['predictions_path']))
    elif command != 'test':
        raise ValueError(f'Unknown command: {command}')


def dump_config(command, args):
    with open(os.path.join(args['experiments'], args['exp_name'], f'config_{command}.json'), 'w') as f:
        json.dump(args, f, indent=4)
