#!/bin/bash

# Slurm options, 8 workers, 1 GPU
#SBATCH -c 8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=logs/creating_intermediate_layer_representations_1280_images_log.out

# Loading the required module
source /etc/profile
module load anaconda/Python-ML-2024b
# custom edits for our specific Python environment
source activate distrib-shift

# Run the script
echo "Creating intermediate layer representations for 10 batches, saving to /home/gridsan/hmartinez/distribution-shift/interpretability/representations.pt"
echo "Using the train loader, first 10 batches"
python /home/gridsan/hmartinez/distribution-shift/interpretability/A1.py