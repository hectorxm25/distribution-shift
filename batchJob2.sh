#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/TRADES_varying_beta6-10normInf_log.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift


echo "Running 6betaNormInf experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/6betaNormInf.pt --beta=6 --constraint=inf --attack_type=trades --eps=0.031

echo "Running 10betaNormInf experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/10betaNormInf.pt --beta=10 --constraint=inf --attack_type=trades --eps=0.031
