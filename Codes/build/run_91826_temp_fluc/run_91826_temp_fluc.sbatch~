#!/bin/bash
# Job name:
#SBATCH --job-name=run_32326_temp_fluc_L_4_5
#
# Partition:
#SBATCH --partition=cmt
#
# Processors:
#SBATCH --ntasks=128
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=1-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxr190099@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o run_32326_temp_fluc_L_4_5.o%j
#------------------------------------------------------

echo "Starting job run_32326_temp_fluc_L_4_5"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=run_32326_temp_fluc_L_4_5.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job run_32326_temp_fluc_L_4_5 complete"
