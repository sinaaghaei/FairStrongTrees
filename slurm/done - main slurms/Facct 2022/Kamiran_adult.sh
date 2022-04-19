#!/bin/bash
sbatch --array 0-99 slurm_FlowOCT_kamiran_10800_adult.sh
sbatch --array 100-199 slurm_FlowOCT_kamiran_10800_adult.sh
sbatch --array 200-299 slurm_FlowOCT_kamiran_10800_adult.sh
sbatch --array 300-399 slurm_FlowOCT_kamiran_10800_adult.sh
sbatch --array 400-499 slurm_FlowOCT_kamiran_10800_adult.sh
sbatch --array 500-599 slurm_FlowOCT_kamiran_10800_adult.sh
sbatch --array 600-699 slurm_FlowOCT_kamiran_10800_adult.sh
sbatch --array 700-839 slurm_FlowOCT_kamiran_10800_adult.sh
