<<<<<<< HEAD
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=dvec_SCOTUS
#SBATCH --output=scotusprint.out

python dvector_SCOTUS.py
