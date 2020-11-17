#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0
NETID=jt2565
source activate /scratch/${NETID}/capstone/newenv
python dvector_create_SCOTUS.py