#aprun -n 4 -N 4 -j 2 -d 32 -cc depth -e OMP_NUM_THREADS=32 -e KMP_BLOCKTIME=0 python mtnetwork-training-single-epoch.py --num_threads=32 --device cpu



#aprun -n 2 -N 2 -j 1 -d 32 -cc depth -e OMP_NUM_THREADS=32 -e KMP_BLOCKTIME=0 python mtnetwork-training-single-epoch.py --num_threads=32 --device cpu
aprun -n 2 -N 2 -j 1 -d 32 -cc depth -e OMP_NUM_THREADS=32 -e KMP_BLOCKTIME=0 python new-sol-mtnetwork-training-single-epoch.py --num_threads=32 --device cpu


# This one works
#aprun -n 1 -N 1 -j 1 -d 64 -cc depth -e OMP_NUM_THREADS=64 -e KMP_BLOCKTIME=0 python mtnetwork-training-single-epoch.py --num_threads=64 --device cpu
#aprun -n 1 -N 1 -j 1 -d 64 -cc depth -e OMP_NUM_THREADS=64 -e KMP_BLOCKTIME=0 python new-sol-mtnetwork-training-single-epoch.py --num_threads=64 --device cpu
