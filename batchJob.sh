#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/TRADES_varying_beta_norm2_log.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift

# Run the script
echo "Running 0.1betaNorm2 experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/0.1betaNorm2 --beta=0.1 --constraint=2 --attack_type=trades --eps=0.5

echo "Running 1betaNorm2 experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/1betaNorm2 --beta=1 --constraint=2 --attack_type=trades --eps=0.5

echo "Running 6betaNorm2 experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/6betaNorm2 --beta=6 --constraint=2 --attack_type=trades --eps=0.5

echo "Running 10betaNorm2 experiment"
python /home/gridsan/hmartinez/distribution-shift/adversarial/train.py --output_dir=/home/gridsan/hmartinez/distribution-shift/models/adversarial/TRADES/10betaNorm2 --beta=10 --constraint=2 --attack_type=trades --eps=0.5
