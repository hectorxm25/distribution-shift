#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/twice_natural_training_log_eps0_CIFAR10_resnet18.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift

# Run the script
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py