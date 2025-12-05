#!/bin/bash
#SBATCH --job-name=my_h200_job
#SBATCH --partition=ice-gpu      # H200 GPU partition (PACE-ICE)
#SBATCH --gres=gpu:h200:1            # Number of H200 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # Adjust CPU count as needed
#SBATCH --mem=64G                    # Adjust memory as needed
#SBATCH --time=4:00:00              # Max runtime (hh:mm:ss)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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
python train.py
