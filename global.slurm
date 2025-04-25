#!/bin/bash

#SBATCH --job-name=CNN_Forward_Pass_Global   # Job name
#SBATCH --time=00:10:00             # Max runtime (10 minutes)
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --mem=1G                    # Memory request (4GB)
#SBATCH --output=forward_pass_execution_global_%j.out  # Output log file
#SBATCH --gres=gpu:a100:1            # Request 1 RTX GPU
#SBATCH --partition=gpu             # Use the GPU partition

# Load CUDA module (if required)

# Compile CUDA code
nvcc inference_global.cu -o global

# Run the executable
./global
