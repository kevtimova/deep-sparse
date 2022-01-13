import argparse
import json
import os
import random
import subprocess
from importlib import import_module

import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

from utils import set_random_seed, get_dataset, np, my_iterator_val

def define_args():
    # Define arguments
    parser = argparse.ArgumentParser(description='Computing codes from pre-trained encoder.')
    parser.add_argument('--name', type=str, default='',
                        help='Name of experiment.')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='Random seed.')
    parser.add_argument('--outdir', default='./results/', type=str,
                        help='Path to the directory that contains the outputs.')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Name of the dataset (options: MNIST | imagenet_LCN)')
    parser.add_argument('--datadir', default='../sample_data', type=str,
                        help='Path to the directory that contains the data.')
    parser.add_argument('--batch_size', type=int, default=250, metavar='N',
                        help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='Number of workers.')
    parser.add_argument('--code_dim', type=int, default=128,
                        help='Code dimension.')
    parser.add_argument('--im_size', type=int, default=28,
                        help='Image input size.')
    parser.add_argument('--patch_size', type=int, default=0,
                        help='Patch size to sample after rescaling to im_size (0 if no patch sampling).')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Whether to run code on GPU (default: run on CPU).')
    parser.add_argument('--encoder', default='lista_encoder', type=str,
                        help='Encoder architecture.')
    parser.add_argument('--num_iter_LISTA', type=int, default=3,
                        help='Number of LISTA iterations.')
    parser.add_argument('--pretrained_path_enc', default='', type=str,
                        help='Path to the pre-trained encoder or decoder.')

    # Get arguments
    args = parser.parse_args()
    return args

def compute_codes(args):
    # Logistics
    args.use_encoder = len(args.pretrained_path_enc) > 0

    # Experiment name
    if args.name == '':
        model_name = args.pretrained_path_enc.split('/')[-1].split('.pth')[0]
        args.name = f'{model_name}_codes'
    print('\nComputing: {}\n'.format(args.name))
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    device = torch.device("cuda" if args.cuda else "cpu")

    # Working directory
    outdir = lambda dirname: os.path.join(args.outdir, dirname)
    if not os.path.exists(outdir('codes')):
        os.mkdir(outdir('codes'))

    # Experiment directories
    out_dir = os.path.join(outdir('codes'), args.name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Random seed
    set_random_seed(args.seed, torch, np, random, args.cuda)

    # Training and test data
    dataset_train = get_dataset(args.dataset, args.datadir,
                                train=True, im_size=args.im_size,
                                patch_size=args.patch_size)
    dataset_test = get_dataset(args.dataset, args.datadir,
                               train=False, im_size=args.im_size,
                               patch_size=args.patch_size)

    # Fix order of training and test data
    data_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.cuda, shuffle=False, drop_last=False)
    data_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                           pin_memory=args.cuda, shuffle=False, drop_last=False)

    # Get data information
    args.n_channels = dataset_test.__getitem__(0)[0].shape[0]

    # Load encoder
    encoder = getattr(import_module('models.{}'.format(args.encoder)), 'Encoder')(args).to(device)
    encoder.load_pretrained(args.pretrained_path_enc, freeze=True)
    encoder.eval()

    for data in [{'batches': data_train, 'split': 'train'},
                 {'batches':  data_test, 'split':  'test'}]:
        split = data['split']
        codes_to_save = None
        targets = None

        for batch, batch_info, should in my_iterator_val(args=args, data=data['batches'], log_interval=1):

            # Logistics
            y = batch['X'].to(device)
            target = batch['target']

            # Compute the codes using amortized inference
            Zs = encoder(y)

            # Concatenate codes
            if codes_to_save is None:
                targets = target
                codes_to_save = Zs
            else:
                codes_to_save = torch.cat([codes_to_save, Zs], dim=0)
                targets = torch.cat([targets, target], dim=0)

        # Save codes and targets
        np.save(os.path.join(out_dir, f'{args.dataset}_{split}_codes.npy'), codes_to_save.cpu().numpy())
        np.save(os.path.join(out_dir, f'{args.dataset}_{split}_targets.npy'), targets.cpu().numpy())
        if 'train' in split:
            # Compute the mean and std of the codes for the training data
            np.save(os.path.join(out_dir, f'{args.dataset}_codes_mean.npy'), codes_to_save.mean().cpu().numpy())
            np.save(os.path.join(out_dir, f'{args.dataset}_codes_std.npy'), codes_to_save.std().cpu().numpy())

        # Final message
        final_msg = f'Finished computing {split} codes.'
        print(final_msg)


if __name__ == '__main__':
    # Get arguments
    args = define_args()

    # Save git info
    args.git_head = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    args.git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    compute_codes(args)
