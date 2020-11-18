#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=dvec_SCOTUS
#SBATCH --output=scotusprint.out

python dvector_create_SCOTUS.py
