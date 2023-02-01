#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A CSC249ADCD08
#PBS -l filesystems=home:grand:eagle

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=64
NDEPTH=1
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_polaris.py /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/configs/config_1521500_monoclinic.txt
