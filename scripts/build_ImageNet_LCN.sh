#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --output=build_imagenet_LCN.out

INPUT_DATASET_PATH="/path/to/imagenet/directory"
OUTPUT_DATASET_PATH="/path/to/output/directory"

python -u imagenet_LCN_patches.py \
--batch_size "250" \
--cuda \
--datadir $INPUT_DATASET_PATH \
--im_size "256" \
--n_test "20000" \
--n_train "200000" \
--n_val "20000" \
--num_workers "4" \
--outdir $OUTPUT_DATASET_PATH \
--patch_size "52" \
--patch_type "random" \
--seed "31" \
--std_threshold "0.5"
