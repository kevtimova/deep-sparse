import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_dataset, FixedSubsetSampler, get_gaussian_filter, LocalContrastNorm

def define_args():
    # Arguments
    parser = argparse.ArgumentParser(description='Generate locally contrast normalized ImageNet patches.')
    parser.add_argument('--datadir', type=str, default='',
                        help='Path to original Imagenet dataset.')
    parser.add_argument('--outdir', type=str, default='',
                        help='Path for new LCN dataset.')
    parser.add_argument('--seed', type=int, default=29,
                        help='random seed (default: 29)')
    parser.add_argument('--n_train', type=int, default=200000,
                        help='Number of training samples.')
    parser.add_argument('--n_val', type=int, default=20000,
                        help='Number of validation samples.')
    parser.add_argument('--n_test', type=int, default=20000,
                        help='Number of test samples.')
    parser.add_argument('--im_size', type=int, default=256,
                        help='Image input size.')
    parser.add_argument('--patch_size', type=int, default=52,
                        help='Patch size to sample after rescaling to im_size.')
    parser.add_argument('--patch_type', type=str, default='random',
                        help='Random or center patch.')
    parser.add_argument('--batch_size', type=int, default=250,
                        help='Batch size for reading data.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Whether to run code on GPU (default: run on CPU).')
    parser.add_argument('--gaussian_filter_radius', type=int, default=13,
                        help='Size of Gaussian Filter.')
    parser.add_argument('--gaussian_filter_sigma', type=float, default=5,
                        help='Std of Gaussian Filter.')
    parser.add_argument('--std_threshold', type=float, default=0.5,
                        help='Threshold for std of selected patches.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of passes through the data.')
    # Parse arguments
    args = parser.parse_args()
    return args

def generate_LCN_patches(data, gaussian_filter, args):

    # Compute the LCN train patches
    lcn_patches = None
    lcn_patches_mean = None
    lcn_patches_std = None
    n_patches = 0
    break_loops = False
    for epoch in range(args.epochs):
        for img, _ in data:
            img = img.to(device)

            # Whitening
            img, img_mean, img_std = LocalContrastNorm(img, gaussian_filter)
            selected_patches = img.std((1, 2, 3)) >= args.std_threshold

            if lcn_patches is None:
                lcn_patches = img[selected_patches].cpu().numpy()
                lcn_patches_mean = img_mean[selected_patches].cpu().numpy()
                lcn_patches_std = img_std[selected_patches].cpu().numpy()
            else:
                lcn_patches = np.append(lcn_patches, img[selected_patches].cpu().numpy(), axis=0)
                lcn_patches_mean = np.append(lcn_patches_mean, img_mean[selected_patches].cpu().numpy(), axis=0)
                lcn_patches_std = np.append(lcn_patches_std, img_std[selected_patches].cpu().numpy(), axis=0)

            n_patches += sum(selected_patches)
            print(f'Epoch {epoch} selected: {sum(selected_patches)}. Total: {n_patches}.')
            if n_patches >= args.n_train + args.n_val + args.n_test:
                break_loops = True
                print(f'Got all the patches during epoch {epoch + 0}!')
                break

        if break_loops:
            break

    # Save train patches
    training = args.n_train + args.n_val
    np.save(os.path.join(args.outdir, f'imagenet_LCN_patches_train.npy'), lcn_patches[:training])
    np.save(os.path.join(args.outdir, f'imagenet_LCN_patches_train_mean.npy'), lcn_patches_mean[:training])
    np.save(os.path.join(args.outdir, f'imagenet_LCN_patches_train_std.npy'), lcn_patches_std[:training])
    print(f'Saved {len(lcn_patches[:training])} train LCN patches.')

    # Save training and validation splits
    np.save(os.path.join(args.outdir, f'imagenet_LCN_train.npy'), [*range(args.n_train)])
    np.save(os.path.join(args.outdir, f'imagenet_LCN_val.npy'), [*range(args.n_train, args.n_train + args.n_val)])

    # Save test patches
    np.save(os.path.join(args.outdir, f'imagenet_LCN_patches_test.npy'), lcn_patches[training:(training + args.n_test)])
    np.save(os.path.join(args.outdir, f'imagenet_LCN_patches_test_mean.npy'), lcn_patches_mean[training:(training + args.n_test)])
    np.save(os.path.join(args.outdir, f'imagenet_LCN_patches_test_std.npy'), lcn_patches_std[training:(training + args.n_test)])
    print(f'Saved {len(lcn_patches[training:(training + args.n_test)])} test LCN patches.')


if __name__ == '__main__':
    # Get arguments
    args = define_args()

    # Save git info
    args.git_head = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    args.git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')

    # Logistics
    device = torch.device("cuda" if args.cuda else "cpu")
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Set random seed
    np.random.seed(args.seed)

    # Read training data
    dataset_start_time = time.time()
    dataset = get_dataset(dataset_name='imagenet', datadir=args.datadir,
                          train=True, im_size=args.im_size,
                          patch_size=args.patch_size, patch_type=args.patch_type)
    n_channels = dataset.__getitem__(0)[0].shape[0]

    # Training and test splits
    dataset_permutation = np.random.permutation(len(dataset)).tolist()

    # Load data
    dataset_sampler = FixedSubsetSampler(dataset_permutation)
    data = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=args.cuda, shuffle=False, sampler=dataset_sampler, drop_last=False)
    dataset_end_time = time.time()
    print(f"Dataset loading time: {dataset_end_time - dataset_start_time:.1f} \n")

    # Gaussian filter
    gaussian_filter = get_gaussian_filter(n_channels, device,
                                          radius=args.gaussian_filter_radius,
                                          sigma=args.gaussian_filter_sigma)

    # Generate patches
    generate_LCN_patches(data, gaussian_filter, args)
