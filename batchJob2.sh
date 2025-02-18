#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/mask_training_natural_and_adversarial_labels_eps=0.031*25_step-size=0.1log.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift


echo "Running Mask Training, natural and adversarial labels, eps=0.031*25, step-size=0.1"
python /home/gridsan/hmartinez/distribution-shift/adversarial/run_experiments.py
