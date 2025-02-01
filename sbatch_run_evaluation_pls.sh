#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --job-name=spin_eval
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --partition=frink
#SBATCH -o output_%j.txt                     # Standard output file
#SBATCH -e error_%j.txt                      # Standard error file
#SBATCH --mail-user=yun.hy@northeastern.edu  # Email
#SBATCH --mail-type=ALL                      # Type of email notifications

# Your program/command here
module purge
# module load explorer anaconda3/2024.06
module load anaconda3/3.6

# source activate base
source activate MedLitSpin
# conda activate MedLitSpin

conda info

bash run_evaluation_pls.sh

conda deactivate
