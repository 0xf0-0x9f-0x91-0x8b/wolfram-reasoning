#!/bin/bash
#SBATCH --job-name=random
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4    # one per GPU
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=4:00:00              # Max runtime (hh:mm:ss)
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

# ---------------------------
#  Load environment
# ---------------------------

# Activate your environment
source "./start.sh" # or "conda activate myenv"

# ---------------------------
#  Runtime info
# ---------------------------
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Running on host: $(hostname)"
echo "GPUs allocated:"
nvidia-smi

# ---------------------------
#  Your command
# ---------------------------

