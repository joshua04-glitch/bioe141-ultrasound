#!/bin/bash
#SBATCH --job-name=cactus
#SBATCH --output=logs/cactus_%j.out
#SBATCH --error=logs/cactus_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load CUDA
module load cuda

# Activate virtual environment
#source /home/users/joshua04/141/torch-gpu/bin/activate
source /home/users/joshua04/141/torch-gpu/bin/activate
# Go to the directory that contains ML train.py
cd /home/users/joshua04/141

# Make this directory visible to Python
export PYTHONPATH=$(pwd)

# Run the converted notebook code
#python "ML train.py" --epochs 40 --unfreeze_epoch 5
python "cactus training.py"
