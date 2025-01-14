#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/regular_mask_accuracy_eps=0.031*25_step-size=0.1log.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift


echo "Running Regular Mask Generation, eps=0.031*25, step-size=0.1"
python /home/gridsan/hmartinez/distribution-shift/adversarial/mask_generation.py
