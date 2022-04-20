#!/bin/bash
sbatch --array 0-99 slurm_FlowOCT_kamiran_warm_10800_limited-adult-2_calibration_mode_1.sh
sbatch --array 100-199 slurm_FlowOCT_kamiran_warm_10800_limited-adult-2_calibration_mode_1.sh
sbatch --array 200-279 slurm_FlowOCT_kamiran_warm_10800_limited-adult-2_calibration_mode_1.sh
