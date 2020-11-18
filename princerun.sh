#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=5GB
#SBATCH --job-name=dvec_SCOTUS

##module purge
##module load anaconda3/5.3.1
##module load cuda/10.0.130
##module load gcc/6.3.0
##source activate /scratch/jt2565/capston/newenv
##cd ~/PyTorch_Speaker_Verification
python dvector_create_SCOTUS.py
