import argparse
import json
import os
import random
import subprocess
import time
from importlib import import_module

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

from utils import get_dataset, set_random_seed, FixedSubsetSampler, my_iterator_val, np, \
    add_noise_to_img, PSNR, L0, dewhiten, inverse_transform, save_img, img_grid

def define_args():
    # Define arguments
    parser = argparse.ArgumentParser(description='Evaluation: denoising.')
    parser.add_argument('--name', type=str, default='',
                        help='Name of experiment.')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='Random seed.')
    parser.add_argument('--outdir', default='./results/', type=str,
                        help='Path to the directory that contains the outputs.')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: MNIST)')
    parser.add_argument('--datadir', default='../sample_data', type=str,
                        help='Path to the directory that contains the data.')
    parser.add_argument('--n_test_samples', type=int, default=10000, metavar='N',
                        help='Number of validation samples for the model.')
    parser.add_argument('--batch_size', type=int, default=250, metavar='N',
                        help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='Number of workers.')
    parser.add_argument('--additive_noise', type=float, default=0,
                        help='Level of noise for input.')
    parser.add_argument('--decoder', default='linear_dictionary', type=str,
                        help='Architecture of pre-trained decoder.')
    parser.add_argument('--code_dim', type=int, default=128,
                        help='Code dimension.')
    parser.add_argument('--pretrained_path_dec', default='', type=str,
                        help='Path to the pre-trained encoder or decoder.')
    parser.add_argument('--im_size', type=int, default=28,
                        help='Image input size.')
    parser.add_argument('--patch_size', type=int, default=0,
                        help='Patch size to sample before rescaling to im_size (0 if no patch sampling).')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Whether to run code on GPU (default: run on CPU).')
    parser.add_argument('--encoder', default='fc_encoder', type=str,
                        help='Encoder architecture.')
    parser.add_argument('--num_iter_LISTA', type=int, default=0,
                        help='Number of LISTA iterations.')
    parser.add_argument('--pretrained_path_enc', default='', type=str,
                        help='Path to the pre-trained encoder or decoder.')
    parser.add_argument('--hidden_dim', type=int, default=128, metavar='N',
                        help='Hidden dimension for multi-layer decoder.')

    # Get arguments
    args = parser.parse_args()
    return args

def eval_denoising(args):
    # Logistics
    whitening = args.dataset == 'imagenet_LCN'
    dataset_eval_indices = args.__dict__.pop('dataset_test_indices', None)

    # Experiment name
    if args.name == '':
        model_name = args.pretrained_path_enc.split('/')[-1].split('.pth')[0]
        args.name = f'{model_name}_seed_{args.seed}_n{args.additive_noise}'
        args.name = '{}_{}'.format(args.name, str(int(time.time())))
    print('\nEval experiment: {}\n'.format(args.name))
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    device = torch.device("cuda" if args.cuda else "cpu")

    # Working directory
    outdir = lambda dirname: os.path.join(args.outdir, dirname)

    # Experiment directories
    img_dir = os.path.join(outdir('imgs'), args.name)
    os.mkdir(img_dir)

    # Tensorboard support. To run: tensorboard --logdir <args.outdir>/logs
    experiment_logs_dir = outdir('logs') + '/{}'.format(args.name)
    os.mkdir(experiment_logs_dir)
    writer = SummaryWriter(log_dir=experiment_logs_dir)

    # Random seed
    set_random_seed(args.seed, torch, np, random, args.cuda)

    # Evaluation data
    dataset_eval = get_dataset(args.dataset, args.datadir,
                               train=False, im_size=args.im_size,
                               patch_size=args.patch_size)
    if dataset_eval_indices is None:
        dataset_eval_indices = [*range(len(dataset_eval))][:args.n_test_samples]

    # Order of elements in valudation set can be fixed
    dataset_eval_sampler = FixedSubsetSampler(dataset_eval_indices)
    data_eval = DataLoader(dataset_eval, batch_size=args.batch_size, num_workers=args.num_workers,
                           pin_memory=args.cuda, shuffle=False, sampler=dataset_eval_sampler, drop_last=True)

    # Get data information
    n_batches_val = len(data_eval)
    log_viz_interval = 1
    args.n_channels = dataset_eval.__getitem__(0)[0].shape[0]

    # Load decoder
    decoder = getattr(import_module('models.{}'.format(args.decoder)), 'Decoder')(args).to(device)
    decoder.load_pretrained(args.pretrained_path_dec, freeze=True)
    decoder.eval()

    # Load encoder
    encoder = getattr(import_module('models.{}'.format(args.encoder)), 'Encoder')(args).to(device)
    encoder.load_pretrained(args.pretrained_path_enc, freeze=True)
    encoder.eval()

    # Evaluation data loop
    psnr_orig_aggr = 0
    psnr_noisy_img_aggr = 0
    psnr_noisy_rec_aggr = 0
    L0_orig_aggr = 0
    L0_noisy_aggr = 0
    n_samples = 0
    args.epoch = 0

    for batch, batch_info, should in my_iterator_val(args, data_eval, log_viz_interval):

        # Logistics
        batch_id = batch['batch_id']
        y = batch['X'].to(device)
        batch_size = batch_info['size']

        # Gererate noise for inputs
        y_noisy = add_noise_to_img(y, args.additive_noise, torch)

        # Whitening
        y_orig_mean, y_orig_std = None, None
        y_noisy_mean, y_noisy_std = None, None
        if args.dataset == 'imagenet_LCN':
            y_orig_mean, y_orig_std = batch['extra']
            y_orig_mean, y_orig_std = y_orig_mean.to(device), y_orig_std.to(device)
            y_noisy_mean, y_noisy_std = y_orig_mean, y_orig_std

        # Compute PSNR between original image and noisy image
        psnr_noisy_img = PSNR(y, y_noisy, args.dataset,
                              tar_sample_mean=y_orig_mean, tar_sample_std=y_orig_std,
                              pred_sample_mean=y_noisy_mean, pred_sample_std=y_noisy_std)


        # Compute the Zs for the original and noisy data uding amortized inference
        Zs_orig = encoder(y)
        Zs_noisy = encoder(y_noisy)

        # L0 of codes
        l0_orig = L0(Zs_orig)
        l0_noisy = L0(Zs_noisy)

        # Reconstructions
        y_hat_orig = decoder(Zs_orig)
        y_hat_noisy = decoder(Zs_noisy)

        # PSNR
        psnr_orig = PSNR(y, y_hat_orig, args.dataset,
                         tar_sample_mean=y_orig_mean, tar_sample_std=y_orig_std)
        psnr_noisy_rec = PSNR(y, y_hat_noisy, args.dataset,
                          tar_sample_mean=y_orig_mean, tar_sample_std=y_orig_std,
                          pred_sample_mean=y_noisy_mean, pred_sample_std=y_noisy_std)

        # De-whiten
        if whitening:
            y = dewhiten(y, y_orig_mean, y_orig_std)
            y_hat_orig = dewhiten(y_hat_orig, y_orig_mean, y_orig_std)
            y_noisy = dewhiten(y_noisy, y_noisy_mean, y_noisy_std)
            y_hat_noisy = dewhiten(y_hat_noisy, y_noisy_mean, y_noisy_std)

        # Log PSNR
        writer.add_scalar('psnr_orig', psnr_orig.item(), batch_id)
        writer.add_scalar('psnr_noisy_img', psnr_noisy_img.item(), batch_id)
        writer.add_scalar('psnr_noisy_rec', psnr_noisy_rec.item(), batch_id)
        writer.add_scalar('L0_orig', l0_orig.item(), batch_id)
        writer.add_scalar('L0_noisy', l0_noisy.item(), batch_id)
        psnr_orig_aggr += psnr_orig * batch_size
        psnr_noisy_img_aggr += psnr_noisy_img * batch_size
        psnr_noisy_rec_aggr += psnr_noisy_rec * batch_size
        L0_orig_aggr += l0_orig * batch_size
        L0_noisy_aggr += l0_noisy * batch_size
        n_samples += batch_size

        # Log targets and reconstructions
        if should['log_val_imgs']:
            y = inverse_transform(y[:16], args.dataset).clamp_(min=0, max=1)
            y_noisy = inverse_transform(y_noisy[:16], args.dataset).clamp_(min=0, max=1)
            y_hat_orig = inverse_transform(y_hat_orig[:16], args.dataset).clamp_(min=0, max=1)
            y_hat_noisy = inverse_transform(y_hat_noisy[:16], args.dataset).clamp_(min=0, max=1)
            save_img(y, f'{img_dir}/y_{(batch_id+1):04d}.png', norm=False)
            save_img(y_noisy, f'{img_dir}/y_noisy_{(batch_id+1):04d}.png', norm=False)
            save_img(y_hat_orig, f'{img_dir}/y_hat_orig_{(batch_id+1):04d}.png', norm=False)
            save_img(y_hat_noisy, f'{img_dir}/y_hat_noisy_{(batch_id+1):04d}.png', norm=False)
            writer.add_image(f'y', img_grid(y), batch_id)
            writer.add_image(f'y_noisy', img_grid(y_noisy), batch_id)
            writer.add_image(f'y_hat_orig', img_grid(y_hat_orig), batch_id)
            writer.add_image(f'y_hat_noisy', img_grid(y_hat_noisy), batch_id)

    # Log columns of linear decoder
    n_cols = min(128, args.code_dim)
    cols = decoder.viz_columns(n_cols)
    save_img(cols, f'{img_dir}/lin_dec_cols.png', norm=False, n_rows=int(2 ** (np.log2(n_cols) // 2)))
    writer.add_image(f'lin_dec_cols', img_grid(cols, norm=False), 0)

    # Aggregate PSNR
    args.psnr_orig_aggr = psnr_orig_aggr / n_samples
    args.psnr_noisy_img_aggr = psnr_noisy_img_aggr / n_samples
    args.psnr_noisy_rec_aggr = psnr_noisy_rec_aggr / n_samples
    args.L0_orig_aggr = L0_orig_aggr / n_samples
    args.L0_noisy_aggr = L0_noisy_aggr / n_samples
    final_msg = 'noise {}\tPSNR_orig: {:.3f}\tPSNR_noisy {:.3f}\t' \
                'PSNR_noisy_rec {:.3f}\tL0_orig {:.3f}\tL0_noisy {:.3f}'.format(args.additive_noise,
                                                                         args.psnr_orig_aggr.item(),
                                                                         args.psnr_noisy_img_aggr.item(),
                                                                         args.psnr_noisy_rec_aggr.item(),
                                                                         args.L0_orig_aggr.item(),
                                                                         args.L0_noisy_aggr.item())
    print(final_msg)

    writer.close()


if __name__ == '__main__':
    # Get arguments
    args = define_args()

    # Save git info
    args.git_head = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    args.git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    eval_denoising(args)
