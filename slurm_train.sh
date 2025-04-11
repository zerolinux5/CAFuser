#!/bin/bash
#SBATCH --job-name=cafuser_muses_train
#SBATCH --output=logs/cafuser_muses_train_%j.out
#SBATCH --error=logs/cafuser_muses_train_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
#SBATCH --mail-type=END,FAIL

# exit when any command fails
set -e

# TODO: Replace the path below with the path to your conda installation directory.
# source /path/to/your/conda/etc/profile.d/conda.sh

conda activate cafuser

# TODO: Replace the path below with the path to your DETECTRON2 datasets directory.
# export DETECTRON2_DATASETS=/path/to/your/datasets

# TODO: Change the directory to the root directory of your project.
# cd /path/to/your/project_root

# Run training
python train_net.py \
    --num-gpus 4 \
    --config-file configs/muses/swin/cafuser_swin_tiny_bs8_180k_muses_clre.yaml \
    OUTPUT_DIR output/cafuser_swin_tiny_bs8_180k_muses_clre_train \
    WANDB.NAME cafuser_swin_tiny_bs8_180k_muses_clre