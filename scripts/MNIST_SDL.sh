#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --output=MNIST_SDL.out

DATASET_PATH="/path/to/data/directory"
DATASET_SPLITS_PATH="/path/to/data/train-val-splits"
OUTPUT_PATH="/path/to/output/directory"

python -u main.py \
--batch_size "250" \
--code_dim "128" \
--code_reg "1" \
--cuda \
--data_splits $DATASET_SPLITS_PATH \
--datadir $DATASET_PATH \
--dataset "MNIST" \
--decoder "linear_dictionary" \
--encoder "lista_encoder" \
--epochs "200" \
--FISTA \
--hidden_dim "0" \
--hinge_threshold "0.5" \
--im_size "28" \
--lrt_D "0.001" \
--lrt_E "0.0003" \
--lrt_Z "1" \
--n_steps_inf "200" \
--n_test_samples "10000" \
--n_training_samples "55000" \
--n_val_samples "5000" \
--noise "[1]" \
--norm_decoder "1" \
--num_iter_LISTA "3" \
--num_workers "4" \
--outdir $OUTPUT_PATH \
--patch_size "0" \
--positive_ISTA \
--seed "31" \
--sparsity_reg "0.005" \
--stop_early "0.001" \
--train_decoder \
--train_encoder \
--use_Zs_enc_as_init \
--variance_reg "0" \
--weight_decay_D "0" \
--weight_decay_E "0" \
--weight_decay_E_bias "0" \
--weight_decay_D_bias "0"
