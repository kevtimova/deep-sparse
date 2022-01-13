import collections
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset
from torchvision import datasets as datasets_torch
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, RandomCrop, Grayscale, CenterCrop
from torchvision.utils import save_image, make_grid

MEAN_MNIST = [0.1307]
STD_MNIST  = [0.3081]
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]
MEAN_IMAGENET_GRAY = [0.457]
STD_IMAGENET_GRAY  = [0.259]

class Gauge:
    def __init__(self):
        self.cache = collections.defaultdict(list)

    def add(self, k, v):
        self.cache[k].append(v)

    def get(self, k, clear=False):
        # Get values for key k and delete them
        res = self.cache[k]
        if clear:
            del self.cache[k]
        return res

class FixedSubsetSampler(Sampler):
    r"""Gives a sampler that yields the same set of indices.

        Arguments:
            indices (sequence): a sequence of indices
        """
    def __init__(self, indices):
        self.idx = indices

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

def get_dataset(dataset_name, datadir, train, im_size, patch_size, patch_type='random'):

    # Transformations
    transforms = []

    # Resize & patch
    if dataset_name not in ['codes']:
        transforms.append(Resize(im_size))
        if patch_size > 0:
            if 'random' in patch_type:
                transforms.append(RandomCrop(patch_size))
            elif 'center' in patch_type:
                transforms.append(CenterCrop(patch_size))
            else:
                raise NotImplementedError

    # Normalize
    if dataset_name == 'MNIST':
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=MEAN_MNIST, std=STD_MNIST))
    elif dataset_name == 'imagenet':
        transforms.append(Grayscale())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=MEAN_IMAGENET_GRAY, std=STD_IMAGENET_GRAY))
    elif dataset_name == 'imagenet_LCN':
        pass
    elif dataset_name == 'codes':
        pass
    else:
        raise NotImplementedError

    # Compose transformations
    transforms = Compose(transforms)

    # Read dataset
    if dataset_name == 'MNIST':
        dataset = getattr(datasets_torch, dataset_name)
        dataset = dataset(root=datadir, train=train, download=True, transform=transforms)
    elif dataset_name == 'imagenet':
        split = 'train'
        dataset = datasets_torch.ImageFolder(root=f'{datadir}/{split}', transform=transforms)
    elif dataset_name == 'imagenet_LCN':
        # Read dataset
        split = 'train' if train else 'test'
        img = np.load(os.path.join(datadir, f'imagenet_LCN_patches_{split}.npy'))
        img_mean = np.load(os.path.join(datadir, f'imagenet_LCN_patches_{split}_mean.npy'))
        img_std = np.load(os.path.join(datadir, f'imagenet_LCN_patches_{split}_std.npy'))
        dataset = ImageNetLCN((img, img_mean, img_std))
    elif dataset_name == 'codes':
        # Read dataset
        split = 'train' if train else 'test'
        codes = np.load(os.path.join(datadir, f'MNIST_{split}_codes.npy'))
        targets = np.load(os.path.join(datadir, f'MNIST_{split}_targets.npy'))
        dataset = Codes((codes, targets))
    else:
        raise NotImplementedError
    return dataset

def inverse_transform(X, dataset_name):
    if dataset_name == 'MNIST':
        mean = MEAN_MNIST[0]
        std = STD_MNIST[0]
    elif dataset_name == 'imagenet_LCN':
        mean = MEAN_IMAGENET_GRAY[0]
        std = STD_IMAGENET_GRAY[0]
    else:
        raise NotImplementedError
    return X * std + mean

def ISTA_step(x, alpha, step_size, positive, stop_early):
    z_prox = x.detach().clone()
    # ISTA gradient step followed by a shrinkage step
    with torch.no_grad():
        z_prox.data = soft_threshold(x.detach() - (1 - stop_early) * step_size * x.grad.data,
                                     threshold=(1 - stop_early) * alpha * step_size, positive=positive)
    return nn.Parameter(z_prox)

def soft_threshold(x, threshold, positive):
    # Function which shrinks input by a given threshold
    result = x.sign() * F.relu(x.abs() - threshold, inplace=True)
    if positive:
        return F.relu(result)
    return result

def sqrt_var(x):
    # Computes the unbiased sample variances of input samples of shape (N, d) across the N dimension
    mean_x = x.mean(0)
    v = torch.norm(x - mean_x, p=2, dim=0) / ((x.shape[0] - 1) ** 0.5)
    return v

def ISTA(decoder, y, positive_ISTA, FISTA,
         sparsity_reg, n_steps_inf, lrt_Z,
         use_Zs_enc_as_init, Zs_enc,
         variance_reg, hinge_threshold, code_reg,
         tolerance, training, train_decoder):
    # Housekeeping
    start_time = time.time()
    batch_size = y.shape[0]

    # Turn off gradient for decoder
    decoder.requires_grad_(False)
    decoder.eval()

    # Generate codes
    if use_Zs_enc_as_init:
        Zs = nn.Parameter(Zs_enc.detach().clone())
    else:
        Zs = nn.Parameter(decoder.initZs(batch_size))
    if FISTA:
        aux = nn.Parameter(Zs.detach().clone())
        t_old = 1
    
    # Auxiliary variables for early stopping
    stop_early_dummies = torch.zeros((batch_size, 1), device=Zs.device)
    stop_early_step = torch.zeros((batch_size, 1), device=Zs.device)

    # Inference iterations
    for step in range(n_steps_inf):
        trainable_param = aux if FISTA else Zs
        loss_dict = loss_f(trainable_param, decoder, y,
                           variance_reg, hinge_threshold,
                           code_reg, Zs_enc)
        total_loss = loss_dict['total_loss']

        # Gradient computation for the codes
        trainable_param.grad = None
        total_loss.backward()

        # Keep track of the codes from the previous iteration
        Zs_old = Zs.detach().clone()

        # Gradient and shrinkage step
        Zs = ISTA_step(x=trainable_param,
                       alpha=sparsity_reg,
                       step_size=lrt_Z,
                       positive=positive_ISTA,
                       stop_early=stop_early_dummies)

        # FISTA
        if FISTA:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
            aux = nn.Parameter(Zs.detach() + ((t_old - 1) / t_new) * (Zs.detach() - Zs_old))
            t_old = t_new

        # Log metrics
        stop_early_dummies = stop_early(Zs_old, Zs.detach(), tolerance)

        # Stop early
        stop_early_step += (1 - stop_early_dummies)
        if step < n_steps_inf - 1:
            if stop_early_dummies.sum() == batch_size:
                break
    # Track time
    elapsed_time = time.time() - start_time

    # Count number of total steps
    Zs_steps_mean = stop_early_step.mean()

    # Remove gradient
    Zs = Zs.detach()
    Zs.requires_grad = False

    # Turn on gradient for decoder
    if training and train_decoder:
        decoder.requires_grad_(True)
        decoder.train()

    output = {'Zs': Zs,
              'inf_steps': torch.FloatTensor([Zs_steps_mean]).to(Zs.device),
              'inference_time': torch.FloatTensor([elapsed_time]).to(Zs.device)}
    return output

def loss_f(Zs, decoder, y, variance_reg, hinge_threshold, code_reg, Zs_enc):
    total_loss = 0

    # Reconstruction loss
    y_hat = decoder(Zs)
    rec_loss = MSE(y, y_hat, reduction='sum')
    total_loss += rec_loss

    # Hinge regularization
    if variance_reg > 0:
        hinge_loss = hinge(input=sqrt_var(Zs), threshold=hinge_threshold, reduction='sum')
        total_loss += variance_reg * hinge_loss
    else:
        hinge_loss = hinge(input=sqrt_var(Zs.detach()), threshold=hinge_threshold, reduction='sum')

    # Distance to the encoder's predictions
    code_loss = None
    if code_reg > 0:
        code_loss = MSE(Zs, Zs_enc.detach(), reduction='sum')
        total_loss += code_reg * code_loss

    output = {'total_loss': total_loss, 'rec_loss': rec_loss.detach(), 'y_hat': y_hat.detach(),
              'hinge_loss': hinge_loss.detach()}
    if code_loss is not None:
        output['code_loss'] = code_loss.detach()

    return output

def MSE(target, pred, reduction='sum'):
    assert target.shape == pred.shape
    dims = (1, 2, 3) if len(target.shape) == 4 else 1
    mean_sq_diff = ((target - pred) ** 2).mean(dims)
    if reduction == 'sum':
        return mean_sq_diff.sum()
    elif reduction == 'mean':
        return mean_sq_diff.mean()
    elif reduction == 'none':
        return mean_sq_diff

def PSNR(target, pred, dataset, tar_sample_mean=None, tar_sample_std=None, pred_sample_mean=None, pred_sample_std=None,
         R=1, dummy=1e-4, reduction='mean'):
    with torch.no_grad():
        # Map inputs back to image space
        if tar_sample_mean is not None:
            target = (target * tar_sample_std) + tar_sample_mean
            if pred_sample_mean is not None:
                # Prediction comes from sample different from the target (e.g. in the case of denoising)
                pred = (pred * pred_sample_std) + pred_sample_mean
            else:
                pred = (pred * tar_sample_std) + tar_sample_mean
        target = inverse_transform(target, dataset)
        pred = inverse_transform(pred, dataset)

        # Compute the PSNR
        dims = (1, 2, 3) if len(target.shape) == 4 else 1
        mean_sq_err = ((target - pred)**2).mean(dims)
        mean_sq_err = mean_sq_err + (mean_sq_err == 0).float() * dummy # if 0, fill with dummy -> PSNR of 40 by default
        output = 10*torch.log10(R**2/mean_sq_err)
        if reduction == 'mean':
            return output.mean()
        elif reduction == 'none':
            return output

def L0(z, reduction='mean', grad=False):
    """
    :param z: (B, C) or (B, C, W, H) tensor
    :return: average of proportion of zero elements in each element in batch
    """
    if not(grad):
        z = z.detach()
    assert (len(z.shape) == 2 or len(z.shape) == 4)
    dims = 1 if len(z.shape) == 2 else (1, 2, 3)
    prop_0s_each_sample = (z.abs() == 0).float().mean(dims)
    if reduction == 'sum':
        return prop_0s_each_sample.sum()
    if reduction == 'mean':
        return prop_0s_each_sample.mean()

def L1(z, reduction='mean', grad=False):
    if not(grad):
        z = z.detach()
    if reduction == 'sum':
        return torch.norm(z, p=1, dim=1).sum()
    elif reduction == 'mean':
        return torch.norm(z, p=1, dim=1).mean()

def stop_early(z_old, z_new, tolerance, absolute=False):
    if tolerance == 0:
        device = torch.device("cuda" if z_old.is_cuda else "cpu")
        shape = (z_old.shape[0], 1) if len(z_old.shape) == 2 else (z_old.shape[0], 1, 1, 1)
        return torch.zeros(size=shape, device=device)
    with torch.no_grad():
        code_dim = 1 if len(z_old.shape) == 2 else (1, 2, 3)
        if absolute:
            diff = torch.norm(z_old - z_new, p=2, dim=code_dim) / z_old[0].numel()
        else:
            diff = torch.norm(z_old - z_new, p=2, dim=code_dim) / torch.norm(z_old, p=2, dim=code_dim)
        if len(z_old.shape) == 2:
            return (diff < tolerance).float().unsqueeze(-1)
        else:
            return (diff < tolerance).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def set_random_seed(seed, torch, np, random, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def normalize_kernels(net, radius):
    # Set kernels to have fixed norm equal to radius
    if radius > 0 and net.training:
        for _, module in net.named_modules():
            if type(module) == nn.Linear:
                with torch.no_grad():
                    W = module.weight.data
                    norms = W.norm(p=2, dim=0)
                    mask = norms / radius
                    module.weight.data /= mask
            if type(module) == nn.Conv2d:
                with torch.no_grad():
                    W = module.weight.data
                    norm = W.norm(p=2, dim=[0, 2, 3])
                    module.weight.data /= norm.unsqueeze(0).unsqueeze(2).unsqueeze(2) * (1 / radius)

def save_img(tensor, name, norm, n_rows=16, scale_each=False):
    save_image(tensor, name, nrow=n_rows, padding=5, normalize=norm, pad_value=1, scale_each=scale_each)

def img_grid(tensor, norm=False, scale_each=False, n_rows=16):
    return make_grid(tensor, nrow=n_rows, padding=5, normalize=norm, range=None, scale_each=scale_each, pad_value=1)

def my_iterator(args, data, log_interval):
    for epoch in range(args.epochs):
        for batch_id, (X, extra) in enumerate(data):
            batch = {}
            batch['batch_id'] = batch_id
            batch['X'] = X
            if args.dataset == 'imagenet_LCN':
                batch['extra'] = extra
            if args.dataset == 'MNIST':
                batch['target'] = extra

            batch_info = {}
            batch_info['epoch'] = epoch
            batch_info['size'] = X.shape[0]

            should = {}
            should['epoch_start'] = batch_id == 0  # first batch of epoch
            should['epoch_end'] = batch_id == len(data) - 1  # last batch of epoch
            should['log_train_imgs'] = epoch % log_interval == 0 or epoch == args.epochs - 1
            yield batch, batch_info, should

def my_iterator_val(args, data, log_interval, epoch=0):
    for batch_id, (X, extra) in enumerate(data):
        batch = {}
        batch['batch_id'] = batch_id
        batch['X'] = X
        if args.dataset == 'imagenet_LCN':
            batch['extra'] = extra
        if args.dataset == 'MNIST':
            batch['target'] = extra

        batch_info = {}
        batch_info['size'] = X.shape[0]

        should = {}
        should['val_start'] = batch_id == 0  # first batch of epoch
        should['val_end'] = batch_id == len(data) - 1  # last batch of epoch
        should['log_val_imgs'] = epoch % log_interval == 0 or epoch == args.epochs - 1
        yield batch, batch_info, should

def hinge(input, threshold=1.0, reduction='sum'):
    # Hinge loss implementation
    diff = F.relu(threshold - input)
    diff = diff**2
    if reduction == 'sum':
        loss = diff.sum()
    elif reduction == 'mean':
        loss = diff.mean()
    return loss

def add_noise_to_img(y, noise_level, torch):
    # Noise
    noise = noise_level * torch.randn(y.shape, device=y.device)

    # Add noise to input
    y_noisy = y + noise

    # Normalize noisy image
    return y_noisy

def log_viz(decoder, writer, n_samples, y, y_hat, Zs, stats,
            img_dir, decoder_arch, dataset, viz_type, log_all=False):
    # Log target
    y_img = inverse_transform(y[:n_samples], dataset)
    save_img(y_img, f'{img_dir}/{viz_type}_X.png', norm=False)
    writer.add_image(f'{viz_type}/X', img_grid(y_img))

    # Log reconstructions
    y_hat_img = inverse_transform(y_hat[:n_samples], dataset).clamp_(min=0, max=1)
    save_img(y_hat_img, f'{img_dir}/{viz_type}_X_rec.png', norm=False)
    writer.add_image(f'{viz_type}/X_rec', img_grid(y_hat_img))

    if log_all:
        n_samples = min(256, decoder.code_dim)
        # Log decoder columns
        cols = decoder.viz_columns(n_samples, norm_each=True)
        save_img(cols, f'{img_dir}/{viz_type}_top_layer_norm_each.png',
                 norm=False, n_rows=int(2 ** (np.log2(n_samples) // 2)))
        writer.add_image(f'{viz_type}/top_layer_norm_each', img_grid(cols))
        cols = decoder.viz_columns(n_samples, norm_each=False)
        save_img(cols, f'{img_dir}/{viz_type}_top_layer_norm_all.png',
                 norm=False, n_rows=int(2 ** (np.log2(n_samples) // 2)))
        writer.add_image(f'{viz_type}/top_layer_norm_all', img_grid(cols))

        # Log code activations
        if decoder_arch in ['one_hidden_decoder']:
            recs = inverse_transform(decoder.viz_codes(Zs.detach().max(0)[0], n_samples),
                                     dataset).clamp_(min=0, max=1)
            save_img(recs, f'{img_dir}/{viz_type}_code_act.png',
                     norm=False, n_rows=int(2 ** (np.log2(n_samples) // 2)))
            writer.add_image(f'{viz_type}/code_act', img_grid(recs))

        # Save codes for histogram
        np.save(f'{img_dir}/{viz_type}_codes.npy', Zs.detach().cpu().numpy())
        np.save(f'{img_dir}/{viz_type}_Zs_comp_use.npy', stats['Zs_comp_use'].cpu().numpy())

def anneal_learning_rate(optimizer, epoch, lrt, ratio=0.9, frequency=2):
    """Sets the learning rate to the initial LR multiplied by {ratio} every {frequency} epochs"""
    lrt = lrt * (ratio ** (epoch // frequency)) # adjusted lrt
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrt

def print_final_training_msg(results_file, head, msg_pre, msg_post, noise,
                             best_perf_tr, best_perf_val):
    # Save results to file
    final_file = open(results_file, 'w')
    final_file.write(head)
    msg_eval = f"{str(noise)}\t" \
               f"NA\tNA\tNA\tNA\tNA"
    best_tr = f"{msg_pre}\tBEST TRAIN\t{msg_post}" \
              f"{best_perf_tr.get('inf_steps', -1):.0f}\t" \
              f"{best_perf_tr.get('L0_Z', -1):.3f}\t" \
              f"{best_perf_tr.get('L0_H', -1):.3f}\t" \
              f"{best_perf_tr.get('PSNR', -1):.3f}\t" \
              f"{best_perf_tr.get('epoch', -1)}\t" \
              f"{msg_eval}"
    final_file.write(best_tr + '\n')
    best_val = f"{msg_pre}\tBEST VAL\t{msg_post}" \
               f"{best_perf_val.get('inf_steps', -1):.0f}\t" \
               f"{best_perf_val.get('L0_Z', -1):.3f}\t" \
               f"{best_perf_val.get('L0_H', -1):.3f}\t" \
               f"{best_perf_val.get('PSNR', -1):.3f}\t" \
               f"{best_perf_val.get('epoch', -1)}\t" \
               f"{msg_eval}"
    final_file.write(best_val + '\n')
    final_file.close()

    # Print final message
    final_msg_trn = f"BEST TRAIN\t" \
                    f"inf_steps: {best_perf_tr.get('inf_steps', -1):.0f}\t" \
                    f"L0_Z: {best_perf_tr.get('L0_Z', -1):.2f}\t" \
                    f"L0_H: {best_perf_tr.get('L0_H', -1):.2f}\t" \
                    f"PSNR: {best_perf_tr.get('PSNR', -1):.2f}\t" \
                    f"epoch: {best_perf_tr.get('epoch', -1)}"
    final_msg_val = f"BEST VALID\t" \
                    f"inf_steps: {best_perf_val.get('inf_steps', -1):.0f}\t" \
                    f"L0_Z: {best_perf_val.get('L0_Z', -1):.2f}\t" \
                    f"L0_H: {best_perf_val.get('L0_H', -1):.2f}\t" \
                    f"PSNR: {best_perf_val.get('PSNR', -1):.2f}\t" \
                    f"epoch: {best_perf_val.get('epoch', -1)}"
    print(final_msg_trn + '\n' + final_msg_val)

def print_final_eval_msg(results_file, msg_pre, msg_post, args_eval, best_perf_val):
    # Save results to file
    final_file = open(results_file, 'a')
    eval_stats = f"{args_eval.additive_noise}\t" \
                 f"{args_eval.L0_orig_aggr:.3f}\t" \
                 f"{args_eval.psnr_orig_aggr:.3f}\t" \
                 f"{args_eval.psnr_noisy_img_aggr:.3f}\t" \
                 f"{args_eval.L0_noisy_aggr:.3f}\t" \
                 f"{args_eval.psnr_noisy_rec_aggr:.3f}"
    msg_eval = f"{msg_pre}\tFINAL denoising \t{msg_post}" \
               f"{best_perf_val.get('inf_steps', -1):.0f}\t" \
               f"{best_perf_val.get('L0_Z', -1):.3f}\t" \
               f"{best_perf_val.get('L0_H', -1):.3f}\t" \
               f"{best_perf_val.get('PSNR', -1):.3f}\t" \
               f"{best_perf_val.get('epoch', -1)}\t" \
               f"{eval_stats}"
    final_file.write(msg_eval + '\n')
    final_file.close()

def dewhiten(y, y_mean, y_std):
    return y * y_std + y_mean

def compute_energy(y, y_hat, Zs, sparsity_reg, variance_reg, hinge_threshold, code_reg, Zs_enc):
    # Function computing the energy minimized during inference
    with torch.no_grad():
        # Reconstruction + L1 norm energy
        energy = MSE(y, y_hat, reduction='sum') + sparsity_reg * L1(Zs, reduction='sum')
        # Variance regularization energy
        if variance_reg > 0:
            variance_term = hinge(input=sqrt_var(Zs.detach()), threshold=hinge_threshold, reduction='sum')
            energy += variance_reg * variance_term
        # Encoder code regularization energy
        if code_reg > 0:
            enc_code_term = MSE(Zs, Zs_enc.detach(), reduction='sum')
            energy += code_reg * enc_code_term
        return energy

def get_gaussian_filter(channels, device, radius, sigma, dim=2):
    radius = [radius] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, device=device)
            for size in radius
        ]
    )
    for size, std, mgrid in zip(radius, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel

def LocalContrastNorm(image, gaussian_filter, padding=False):
    """
    INPUTS
    images: torch.Tensor of shape (N, ch, h, w)
    gaussian_filter: gaussian filter of size (ch,radius,radius)
    radius: Gaussian filter size (int), odd
    OUTPUT
    locally contrast normalized images of shape (N, ch, h - 2*(radius -1), w - 2*(radius -1)) or (N, ch, h, m)
    depending on whether padding is used
    Modified from: https://github.com/dibyadas/Visualize-Normalizations/blob/master/LocalContrastNorm.ipynb
    """
    _, ch, radius, _ = gaussian_filter.shape
    if radius % 2 == 0:
        radius = radius + 1
    pad = radius // 2

    # Apply Gaussian filter to original patch
    if padding:
        # (N, ch, h, w)
        filter_out = F.conv2d(input=image, weight=gaussian_filter, padding=radius - 1)[:, :, pad:-pad, pad:-pad]
    else:
        # (N, ch, h - r + 1, w - r + 1)
        filter_out = F.conv2d(input=image, weight=gaussian_filter, padding=0)

    # Center
    if padding:
        # (N, ch, h, w)
        centered_image = image - filter_out
    else:
        # (N, ch, h - r + 1, w - r + 1)
        centered_image = image[:, :, pad:-pad, pad:-pad] - filter_out

    # Variance
    if padding:
        var = F.conv2d(centered_image.pow(2), gaussian_filter, padding=radius - 1)[:, :, pad:-pad, pad:-pad]
    else:
        # (N, ch, h - 2*(r - 1), w - 2*(r - 1))
        var = F.conv2d(centered_image.pow(2), gaussian_filter, padding=0)
    var_pos = var >= 0
    var = var * var_pos + 0 * (var_pos == False)

    # Standard deviation
    st_dev = var.sqrt()
    st_dev_mean = st_dev.mean()
    gr_than_mean = st_dev > st_dev_mean
    st_dev = st_dev * gr_than_mean + st_dev_mean * (gr_than_mean == False)
    gr_than_min = st_dev > 1e-4
    st_dev = st_dev * gr_than_min + 1e-4 * (gr_than_min == False)

    # Divide by std
    if padding:
        new_image = centered_image / st_dev
    else:
        new_image = centered_image[:, :, pad:-pad, pad:-pad] / st_dev

    # Return normalized input and stats
    if padding:
        return new_image, filter_out, st_dev
    else:
        return new_image, filter_out[:, :, pad:-pad, pad:-pad], st_dev

class ImageNetLCN(Dataset):

    def __init__(self, dataset):
        self.img, self.mean, self.std = dataset

    def __getitem__(self, index):
        return self.img[index], (self.mean[index], self.std[index])

    def __len__(self):
        return self.img.shape[0]

class Codes(Dataset):

    def __init__(self, dataset):
        self.codes, self.targets = dataset

    def __getitem__(self, index):
        return self.codes[index], self.targets[index]

    def __len__(self):
        return self.codes.shape[0]
