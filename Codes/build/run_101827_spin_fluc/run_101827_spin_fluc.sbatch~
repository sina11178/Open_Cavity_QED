#!/bin/bash
# Job name:
#SBATCH --job-name=run_91826_temp_fluc
#
# Partition:
#SBATCH --partition=cmt
#
# Processors:
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#
# Wall clock limit:
#SBATCH --time=3-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxr190099@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o run_91826_temp_fluc.o%j
#SBATCH -e run_91826_temp_fluc.e%j
#------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1


echo "Starting job run_91826_temp_fluc"

#module load launcher

#export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
#export LAUNCHER_RMI=SLURM
#export LAUNCHER_JOB_FILE=run_91826_temp_fluc.task
#export LAUNCHER_SCHED=interleaved

#$LAUNCHER_DIR/paramrun

#python infinitesmal_temp_fluc_dense.py


RUN_NAME=run_91826_temp_fluc
PROJECT_NAME=Open_Cavity_QED/Codes
SCRATCH=$HOME/scratch
SCRATCH_DIR=$SCRATCH/$RUN_NAME/b0
LOCAL_DIR=$HOME/$PROJECT_NAME/build
EXEC=infinitesmal_temp_fluc_dense.py

mkdir -p $SCRATCH_DIR
cp $LOCAL_DIR/$RUN_NAME/$EXEC $SCRATCH_DIR/
cd $SCRATCH_DIR

echo "Starting job $RUN_NAME"
time python ./$EXEC
echo "Job $RUN_NAME complete"


echo "Job run_91826_temp_fluc complete"
