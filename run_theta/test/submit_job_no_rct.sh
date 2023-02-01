#!/bin/bash
#COBALT -n 1 
#COBALT -t 60 
#COBALT -A CSC249ADCD08 
#COBALT -q debug-flat-quad

module load conda/2021-09-22
export OMP_NUM_THREADS=64
aprun -n 1 -N 1 -cc none python /home/twang3/myWork/test/mtnetwork-training-single-epoch.py -i0
