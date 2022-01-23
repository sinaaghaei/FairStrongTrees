#!/bin/bash
sbatch --array 0-99 slurm_FlowOCT_kamiran_10800_compas-limited.sh
sbatch --array 100-199 slurm_FlowOCT_kamiran_10800_compas-limited.sh
sbatch --array 200-279 slurm_FlowOCT_kamiran_10800_compas-limited.sh
