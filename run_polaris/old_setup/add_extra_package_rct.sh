conda install matplotlib
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install scikit-learn
pip install mpi4py
#The conda installation of mpi4py is not correct! It does not generate a libomp.so, but pip can!
pip install h5py
