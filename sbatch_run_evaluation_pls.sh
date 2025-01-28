#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --job-name=spin_eval
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:4
#SBATCH -o output_%j.txt                     # Standard output file
#SBATCH -e error_%j.txt                      # Standard error file
#SBATCH --mail-user=yun.hy@northeastern.edu  # Email
#SBATCH --mail-type=ALL                      # Type of email notifications

# Your program/command here
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

source activate base
source activate MedLitSpin
conda activate MedLitSpin

conda info

bash run_evaluation_pls.sh

conda deactivate
