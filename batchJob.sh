#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/TRADES_varying_beta0.1-1normInf_log.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift

# Run the script
echo "Running 0.1betaNormInf experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/0.1betaNormInf.pt --beta=0.1 --constraint=inf --attack_type=trades --eps=0.031

echo "Running 1betaNormInf experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/1betaNormInf.pt --beta=1 --constraint=inf --attack_type=trades --eps=0.031