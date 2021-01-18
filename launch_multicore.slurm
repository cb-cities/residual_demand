#!/bin/bash
#SBATCH -J tokyo
#SBATCH -o a22m.out
#SBATCH -e a22m.err
#SBATCH -t 10:00:00
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40

module purge
module load TACC
module load launcher
module load python3
source $HOME/virtualenvs/cities/bin/activate
export OMP_NUM_THREADS=40

### folder in $SCRATCH to hold output
scratch_folder=$SCRATCH/residual_demand/projects/tokyo_osmnx/simulation_outputs
mkdir -p $scratch_folder
mkdir -p $scratch_folder/log
mkdir -p $scratch_folder/edge_vol
mkdir -p $scratch_folder/trip_info

# Launcher to use multicores on node
export LAUNCHER_WORKDIR=$PWD
export LAUNCHER_PPN=1
export LAUNCHER_JOB_FILE=joblist_a22m
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

# cd $scratch_folder
# tar czf ../t_stats_$(date +"%FT%H%M%S").tar.gz t_stats