#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --job-name=dvcase
#SBATCH --output=bycase.out

python ~/SpeakerVerificationEmbedding/SCOTUS/dvec_bycase.py