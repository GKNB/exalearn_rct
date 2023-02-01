export PYTHONNOUSERSITE=True
export RCT_CONDA_ENV=rct-universal

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
chmod +x $HOME/miniconda.sh
$HOME/miniconda.sh -b -p $HOME/.miniconda3
source $HOME/.miniconda3/bin/activate
conda update -y -n base -c defaults conda
conda create -y -n $RCT_CONDA_ENV python=3.7
conda activate $RCT_CONDA_ENV
conda config --add channels conda-forge

conda install -y apache-libcloud chardet colorama dill future idna \
                 msgpack-python netifaces ntplib parse 'pymongo<4' pyzmq \
                 regex requests setproctitle urllib3

pip install radical.utils radical.gtod radical.saga
pip install radical.pilot radical.entk

radical-stack
