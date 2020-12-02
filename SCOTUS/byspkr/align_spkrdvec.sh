#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=alspkr
#SBATCH --output=al_byskpr.out

python SpeakerVerificationEmbedding/SCOTUS/align_byspkr.py
echo completed
