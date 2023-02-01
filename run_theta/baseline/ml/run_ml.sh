#!/bin/bash
#COBALT -n 2
#COBALT -t 01:00:00
#COBALT -A CSC249ADCD08
#COBALT -q debug-flat-quad
#COBALT --attrs mcdram=cache:numa=quad:filesystems=home,theta-fs0

echo "Starting Cobalt job script"

export n_nodes=${COBALT_JOBSIZE}
export n_mpi_ranks_per_node=1
export n_mpi_ranks=$(($n_nodes * $n_mpi_ranks_per_node))
export n_cores_per_rank=$((64 / $n_mpi_ranks_per_node))
export n_threads_per_core=1

module load conda/2021-09-22

# With horovod
aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d $n_cores_per_rank -j $n_threads_per_core -cc depth -e OMP_NUM_THREADS=32 python mtnetwork-training-horovod.py --num_threads=32 --device cpu

# No horovod
#aprun -n 1 -N 1 -d 64 -j 1 -cc depth -e OMP_NUM_THREADS=64 python mtnetwork-training-no-horovod.py
#aprun -n 1 -N 1 -d 16 -j 1 -cc depth -e OMP_NUM_THREADS=16 python mtnetwork-training-no-horovod.py
#aprun -n 1 -N 1 -d 64 -j 1 -cc depth -e OMP_NUM_THREADS=1 python mtnetwork-training-no-horovod.py
#aprun -n 1 -N 1 -d 64 -j 1 -cc depth -e OMP_NUM_THREADS=32 python mtnetwork-training-no-horovod.py
