#!/bin/sh
#SBATCH --job-name=hello-daic
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1GB
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

srun python3 script.py
