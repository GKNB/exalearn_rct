#!/bin/bash
#COBALT -n 1 
#COBALT -t 60 
#COBALT -A CSC249ADCD08 
#COBALT -q debug-flat-quad
#COBALT --attrs mcdram=cache:numa=quad

echo "Starting Cobalt job script"
export n_nodes=$COBALT_JOBSIZE
export n_mpi_ranks_per_node=64
export n_mpi_ranks=$(($n_nodes * $n_mpi_ranks_per_node))

#aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d 1 -j 1 --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/configs/config_1001460_cubic.txt

aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d 1 -j 1 --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/configs/config_1522004_trigonal.txt

#aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d 1 -j 1 --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/configs/config_1001460_cubic.txt
