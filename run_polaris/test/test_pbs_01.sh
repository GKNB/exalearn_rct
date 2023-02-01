#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A CSC249ADCD08

# Change to working directory
echo Working directory is $PBS_O_WORKDIR
cd ${PBS_O_WORKDIR}

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
echo "NUM_PPN = $NUM_PPN"
echo "SAGA_PPN = $SAGA_PPN"

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads /home/twang3/myWork/exalearn_project/run_polaris/test/run_on_compute_node_00.sh
