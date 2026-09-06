#!/bin/bash

# --- Slurm Resource Requests ---
#SBATCH --account=cil_jobs       # Your 200-hour job account
#SBATCH --partition=jobs         # The required queue
#SBATCH --time=04:00:00          # Time limit
#SBATCH --gpus=1                 # Request 1 standard GPU
#SBATCH --output=logs/slurm-%j.out

# Exit immediately if a command fails
set -e

echo "===== SLURM JOB STARTING ====="

# 1. PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Load the required CUDA module (uv installed PyTorch, but the system still needs this)
module load cuda/12.6.0

# 3. Point directly to the virtual environment Python
PROJECT_ROOT=$(pwd)
PYTHON="$PROJECT_ROOT/.venv/bin/python"

# 4. Tell Python to look in the project root for the 'cil' folder
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export PYTHONUNBUFFERED=1
# 5. Run your script using the specific uv Python executable!
# Notice we use $PYTHON instead of just 'python'
srun $PYTHON "$@"

echo "===== SLURM JOB COMPLETE ====="