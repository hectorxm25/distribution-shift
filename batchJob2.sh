#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/random_noise_mask_confidence_superimposed_and_not_log.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift


echo "Running run_random_noise_mask_confidence_experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/run_experiments.py
