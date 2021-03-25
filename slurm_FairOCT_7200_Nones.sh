#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --constraint="xeon-2640v4"
#SBATCH --array=0-9


module load gcc
module load gurobi
module load python

export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}

dataset_enc_list="compas_enc.csv compas_enc.csv compas_enc.csv compas_enc.csv compas_enc.csv compas_enc.csv compas_enc.csv compas_enc.csv compas_enc.csv compas_enc.csv
"
dataset_reg_list="compas.csv compas.csv compas.csv compas.csv compas.csv compas.csv compas.csv compas.csv compas.csv compas.csv
"
depth_list="1 2 1 2 1 2 1 2 1 2
"
sample_list="1 1 2 2 3 3 4 4 5 5
"
fairness_type_list="None None None None None None None None None None
"
bounds_list="0 0 0 0 0 0 0 0 0 0
"
protected_feature_list="race_factor race_factor race_factor race_factor race_factor race_factor race_factor race_factor race_factor race_factor
"
condition_feature_list="gender_factor gender_factor gender_factor gender_factor gender_factor gender_factor gender_factor gender_factor gender_factor gender_factor
"
dataset_enc_list=($dataset_enc_list)
dataset_reg_list=($dataset_reg_list)
depth_list=($depth_list)
sample_list=($sample_list)
fairness_type_list=($fairness_type_list)
bounds_list=($bounds_list)
protected_feature_list=($protected_feature_list)
condition_feature_list=($condition_feature_list)


python FlowOCTReplication.py -r ${dataset_reg_list[$SLURM_ARRAY_TASK_ID]} -f ${dataset_enc_list[$SLURM_ARRAY_TASK_ID]} -d ${depth_list[$SLURM_ARRAY_TASK_ID]} -t 7200 -l 0 -i ${sample_list[$SLURM_ARRAY_TASK_ID]} -c 1 -a ${fairness_type_list[$SLURM_ARRAY_TASK_ID]} -b ${bounds_list[$SLURM_ARRAY_TASK_ID]} -e ${protected_feature_list[$SLURM_ARRAY_TASK_ID]} -g 2 -h ${condition_feature_list[$SLURM_ARRAY_TASK_ID]}
