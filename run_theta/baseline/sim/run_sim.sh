#!/bin/bash
#COBALT -n 1
#COBALT -t 01:00:00
#COBALT -A CSC249ADCD08 
#COBALT -q debug-flat-quad
#COBALT --attrs mcdram=cache:numa=quad

echo "Starting Cobalt job script"

export n_nodes=$COBALT_JOBSIZE
export n_mpi_ranks_per_node=64
export n_mpi_ranks=$(($n_nodes * $n_mpi_ranks_per_node))
export n_cores_per_rank=$((64 / $n_mpi_ranks_per_node))
export n_threads_per_core=1

module load conda/2021-09-22
export LD_LIBRARY_PATH=/home/twang3/g2full_theta/GSASII/bindist:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

#aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d $n_cores_per_rank -j $n_threads_per_core --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1001460_cubic.txt

#aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d $n_cores_per_rank -j $n_threads_per_core --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1522004_trigonal_part1.txt

#aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d $n_cores_per_rank -j $n_threads_per_core --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1522004_trigonal_part2.txt

#aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d $n_cores_per_rank -j $n_threads_per_core --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1531431_tetragonal.txt

aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node -d $n_cores_per_rank -j $n_threads_per_core --cc depth -e OMP_NUM_THREADS=1 python /home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_multi_sym_theta.py /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1001460_cubic.txt /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1522004_trigonal_part1.txt /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1522004_trigonal_part2.txt /home/twang3/myWork/exalearn_project/run_theta/baseline/sim/configs/config_1531431_tetragonal.txt
