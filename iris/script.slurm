#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-12:00:00

# Load required modules
module load Python/3.7.1-GCC-6.2.0

echo "Test Information"
python run.py