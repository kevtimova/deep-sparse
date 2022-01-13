#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --output=ImageNet_VDL-NL.out

DATASET_PATH="/path/to/data/directory"
DATASET_SPLITS_PATH="/path/to/data/train-val-splits"
OUTPUT_PATH="/path/to/output/directory"

python -u main.py \
--anneal_lr_D_freq "30" \
--anneal_lr_D_mult "0.5" \
--batch_size "250" \
--code_dim "256" \
--code_reg "40" \
--cuda \
--data_splits $DATASET_SPLITS_PATH \
--datadir $DATASET_PATH \
--dataset "imagenet_LCN" \
--decoder "one_hidden_decoder" \
--encoder "lista_encoder" \
--epochs "100" \
--FISTA \
--hidden_dim "2048" \
--hinge_threshold "0.5" \
--im_size "28" \
--lrt_D "5e-05" \
--lrt_E "0.0001" \
--lrt_Z "0.5" \
--n_steps_inf "200" \
--n_test_samples "20000" \
--n_training_samples "200000" \
--n_val_samples "20000" \
--noise "[1]" \
--norm_decoder "0" \
--num_iter_LISTA "3" \
--num_workers "4" \
--outdir $OUTPUT_PATH \
--patch_size "0" \
--positive_ISTA \
--seed "31" \
--sparsity_reg "0.01" \
--stop_early "0.001" \
--train_decoder \
--train_encoder \
--use_Zs_enc_as_init \
--variance_reg "10" \
--weight_decay_D "0" \
--weight_decay_E "0" \
--weight_decay_D_bias "0.1" \
--weight_decay_E_bias "0.01"
