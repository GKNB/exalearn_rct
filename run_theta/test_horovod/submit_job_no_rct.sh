#!/bin/bash
#COBALT -n 2
#COBALT -t 30
#COBALT -A CSC249ADCD08
#COBALT -q debug-cache-quad
#COBALT --attrs filesystems=home,theta-fs0

NNODES=${COBALT_JOBSIZE}
NRANKS_PER_NODE=1
NTHREADS_PER_CORE=1
NDEPTH=64

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

module load conda/2021-09-22
aprun -n ${NTOTRANKS} -N ${NRANKS_PER_NODE} -d ${NDEPTH} -j ${NTHREADS_PER_CORE} -cc depth -e OMP_NUM_THREADS=64 \
	python new-sol-mtnetwork-training-single-epoch.py --num_threads=64 --device cpu


#NNODES=${COBALT_JOBSIZE}
#NRANKS_PER_NODE=2
#NTHREADS_PER_CORE=1
#NDEPTH=32
#
#NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
#
#module load conda/2021-09-22
#aprun -n ${NTOTRANKS} -N ${NRANKS_PER_NODE} -d ${NDEPTH} -j ${NTHREADS_PER_CORE} -cc depth -e OMP_NUM_THREADS=32 \
#	python new-sol-mtnetwork-training-single-epoch.py --num_threads=32 --device cpu
