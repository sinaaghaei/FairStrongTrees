#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64GB
#SBATCH --time=05:00:00
#SBATCH --export=NONE


cd /project/vayanou_651/FairStrongTrees/SMT/fair-repair-master/

module load gcc
module load python

export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}


python german_patch.py

