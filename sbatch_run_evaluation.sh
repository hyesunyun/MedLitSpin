#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --job-name=spin_eval
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH -o output_%j.txt                     # Standard output file
#SBATCH -e error_%j.txt                      # Standard error file
#SBATCH --mail-user=yun.hy@northeastern.edu  # Email
#SBATCH --mail-type=ALL                      # Type of email notifications

# Your program/command here
conda activate MedLitSpin

./run_evaluation.sh

conda deactivate