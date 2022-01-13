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
from torch.utils.data.sampler import SubsetRandomSampler

torch.backends.cudnn.benchmark = True

from utils import set_random_seed, get_dataset, FixedSubsetSampler, np

def define_args():
    # Define arguments
    parser = argparse.ArgumentParser(description='Evaluation: classification.')
    parser.add_argument('--name', type=str, default='',
                        help='Name of experiment.')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='Random seed.')
    parser.add_argument('--outdir', default='./results/', type=str,
                        help='Path to the directory that stores the evaluation results.')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='name of the dataset (options: MNIST | codes)')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='Path to the directory that contains the data.')
    parser.add_argument('--data_splits', default='./data', type=str,
                        help='Path to the directory that contains the data splits.')
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes.')
    parser.add_argument('--code_dim', type=int, default=128,
                        help='Code dimension (for LISTA classifier).')
    parser.add_argument('--n_training_samples_per_class', type=int, default=100, metavar='N',
                        help='Number of training samples for the model.')
    parser.add_argument('--n_val_samples', type=int, default=5000, metavar='N',
                        help='Number of validation samples for the model.')
    parser.add_argument('--n_test_samples', type=int, default=10000, metavar='N',
                        help='Number of test samples for the model.')
    parser.add_argument('--batch_size', type=int, default=250, metavar='N',
                        help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='Number of workers.')
    parser.add_argument('--classifier', default='linear_classifier', type=str,
                        help='Architecture of classifier.')
    parser.add_argument('--im_size', type=int, default=28,
                        help='Image input size.')
    parser.add_argument('--patch_size', type=int, default=0,
                        help='Patch size to sample before rescaling to im_size (0 if no patch sampling).')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Whether to run code on GPU (default: run on CPU).')
    parser.add_argument('--n_batches_to_log', type=int, default=20, metavar='N',
                        help='How many batches to log.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for classifier.')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Track top k classification.')
    parser.add_argument('--L1_reg', type=float, default=0,
                        help='Level of L1 regularization for the classifier\'s weights.')
    parser.add_argument('--L2_reg', type=float, default=0,
                        help='Level of L1 regularization for the classifier\'s weights.')

    # Parse arguments
    args = parser.parse_args()
    return args

def eval_class(args):
    # Logistics
    if args.dataset == 'codes':
        try:
            data_source = '-'.join(args.datadir.split('/')[-1].split('_s')[-1].split('_')[1:4])
        except:
            data_source = 'codes'
    else:
        data_source = args.dataset
        if 'lista' in args.classifier:
            data_source += '-' + '-'.join(args.classifier.split('_'))
    if len(args.name) == 0:
        timestamp = str(int(time.time()))
        args.name = f'{data_source}_{timestamp}_s_{args.seed}_' \
                    f'ntr_{args.n_training_samples_per_class}_lr_{args.lr}_' \
                    f'L1_{args.L1_reg}_L2_{args.L2_reg}'
    else:
        timestamp = args.name.split('_s_')[0].split('_')[-1]
    print('\nClassificaion experiment: {}\n'.format(args.name))
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    device = torch.device("cuda" if args.cuda else "cpu")

    # Experiment directories
    outdir = lambda dirname: os.path.join(args.outdir, dirname)
    if not os.path.exists(outdir('classify')):
        os.mkdir(outdir('classify'))
    results_file = os.path.join(outdir('classify'), 'classif_results.tsv')
    model_loc = outdir('checkpoints') + '/{}'.format(args.name) + '.pth'

    # Tensorboard support. To run: tensorboard --logdir <args.outdir>/logs
    logs_dir = outdir('classify') + '/{}'.format(args.name)
    os.mkdir(logs_dir)
    writer = SummaryWriter(log_dir=logs_dir)

    # Random seed
    set_random_seed(args.seed, torch, np, random, args.cuda)

    # Read training data
    dataset_train = get_dataset(args.dataset, args.datadir,
                                train=True, im_size=args.im_size,
                                patch_size=args.patch_size)
    train_indices = np.load(os.path.join(args.data_splits, f'{args.dataset}_train.npy'))

    # Select pre-set number of training elements per class
    train_indices_selected = []
    for class_id in range(args.n_classes):
        selected = dataset_train.targets[train_indices] == class_id
        train_indices_selected = train_indices_selected + list(train_indices[selected][:args.n_training_samples_per_class])
    train_sampler = SubsetRandomSampler(train_indices_selected)
    data_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.cuda, shuffle=False, sampler=train_sampler, drop_last=False)

    # Read validation data
    val_indices = list(np.load(os.path.join(args.data_splits, f'{args.dataset}_val.npy')))[:args.n_val_samples]
    val_sampler = FixedSubsetSampler(val_indices)
    data_val = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=args.cuda, shuffle=False, sampler=val_sampler, drop_last=False)

    # Read test data
    dataset_test = get_dataset(args.dataset, args.datadir,
                               train=False, im_size=args.im_size,
                               patch_size=args.patch_size)
    test_indices = [*range(len(dataset_test))][:args.n_test_samples]
    test_sampler = FixedSubsetSampler(test_indices)
    data_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                           pin_memory=args.cuda, shuffle=False, sampler=test_sampler, drop_last=False)

    # Get data information
    args.n_batches = len(data_train)
    args.log_interval = max(args.n_batches // args.n_batches_to_log, 1) if args.n_batches_to_log > 0 else 0
    args.input_dim = dataset_train.__getitem__(0)[0].flatten().shape[0]

    # Classifier
    classifier = getattr(import_module('models.{}'.format(args.classifier)), 'Classifier')(args).to(device)

    # Classifier training loop
    best_train = {'loss': None, 'ep': None, 'top1': None, f'top{args.top_k}': None}
    best_val = {'loss': None, 'ep': None, 'top1': None, f'top{args.top_k}': None}

    # Training
    for epoch in range(args.epochs):
        # Train classifier
        train_stats = classifier.train(data_train, epoch, args.top_k)
        print('Epoch [{}/{}] Train Loss: {:.6f} Top1: {:.3f} Top{}: {:.3f} N: {}'.format(
            epoch + 1, args.epochs, train_stats['loss'], train_stats['top1'], args.top_k, train_stats[f'top{args.top_k}'],
            train_stats['n_samples']))
        writer.add_scalar('loss_train', train_stats['loss'], epoch)
        writer.add_scalar('top1_train', train_stats['top1'], epoch)
        writer.add_scalar(f'top{args.top_k}_train', train_stats[f'top{args.top_k}'], epoch)

        # Track best training
        if best_train['loss'] is not None:
            if best_train['loss'] > train_stats['loss']:
                best_train['loss'] = train_stats['loss']
                best_train['top1'] = train_stats['top1']
                best_train[f'top{args.top_k}'] = train_stats[f'top{args.top_k}']
                best_train['ep'] = epoch
        else:
            best_train['loss'] = train_stats['loss']
            best_train['top1'] = train_stats['top1']
            best_train[f'top{args.top_k}'] = train_stats[f'top{args.top_k}']
            best_train['ep'] = epoch

        # Validation
        val_stats = classifier.test(data_val, args.top_k)
        print('Epoch [{}/{}] Valid Loss: {:.6f} Top1: {:.3f} Top{}: {:.3f} N: {}'.format(
            epoch + 1, args.epochs, val_stats['loss'], val_stats['top1'], args.top_k, val_stats[f'top{args.top_k}'],
            val_stats['n_samples']))
        writer.add_scalar('loss_val', val_stats['loss'], epoch)
        writer.add_scalar('top1_val', val_stats['top1'], epoch)
        writer.add_scalar(f'top{args.top_k}_val', val_stats[f'top{args.top_k}'], epoch)

        # Track best validation
        if best_val['loss'] is not None:
            if best_val['loss'] > val_stats['loss']:
                best_val['loss'] = val_stats['loss']
                best_val['top1'] = val_stats['top1']
                best_val[f'top{args.top_k}'] = val_stats[f'top{args.top_k}']
                best_val['ep'] = epoch

                # Save best model
                best_val_cl = classifier
                torch.save(classifier.state_dict(), model_loc)
        else:
            best_val['loss'] = val_stats['loss']
            best_val['top1'] = val_stats['top1']
            best_val[f'top{args.top_k}'] = val_stats[f'top{args.top_k}']
            best_val['ep'] = epoch
            best_val_cl = classifier

    # Test performance
    test_stats = best_val_cl.test(data_test, args.top_k)

    # Final message
    tr_msg = '{}\tBEST Training\tLoss: {:.6f}\tTop1: {:.6f}\tTop{}: {:.6f} ep: {}\n'.format(
        args.name, best_train['loss'], best_train['top1'],
        args.top_k, best_train[f'top{args.top_k}'], best_train['ep'] + 1)
    val_msg = '{}\tBEST Validation\tLoss: {:.6f}\tTop1: {:.6f}\tTop{}: {:.6f} ep: {}\n'.format(
        args.name, best_val['loss'], best_val['top1'],
        args.top_k, best_val[f'top{args.top_k}'], best_val['ep'] + 1)
    test_msg = '{}\tBEST Val TEST\tLoss: {:.6f}\tTop1: {:.6f}\tTop{}: {:.6f} ep: {}\n'.format(
        args.name, test_stats['loss'], test_stats['top1'],
        args.top_k, test_stats[f'top{args.top_k}'], best_val['ep'] + 1)
    final_msg = '\n' + tr_msg + val_msg + test_msg
    print(final_msg)

    # Save results
    final = open(results_file, 'a')
    head = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format(data_source, timestamp, args.seed,
                                                 args.n_training_samples_per_class,
                                                 args.lr, args.L1_reg, args.L2_reg)
    train_row = head + 'train\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.format(best_train['loss'], best_train['top1'],
                                                             best_train[f'top{args.top_k}'], best_train['ep'] + 1)
    final.write(train_row)
    val_row = head + 'valid\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.format(best_val['loss'], best_val['top1'],
                                                           best_val[f'top{args.top_k}'], best_val['ep'] + 1)
    final.write(val_row)
    test_row = head + 'test\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.format(test_stats['loss'], test_stats['top1'],
                                                            test_stats[f'top{args.top_k}'], best_val['ep'] + 1)
    final.write(test_row)

    final.close()
    writer.close()


if __name__ == '__main__':
    # Get arguments
    args = define_args()

    # Save git info
    args.git_head = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    args.git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    eval_class(args)
