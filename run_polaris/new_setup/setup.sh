#/bin/bash

module load conda
conda activate /grand/CSC249ADCD08/twang/env/base-clone-polaris

which python
python -V

export RADICAL_PILOT_DBURL=$(cat /home/twang3/myWork/exalearn_project/run_polaris/rp_dburl_polaris)

export RADICAL_LOG_LVL=DEBUG
export RADICAL_PROFILE=TRUE

export RMQ_HOSTNAME=95.217.193.116
export RMQ_PORT=5672
export RMQ_USERNAME=litan
export RMQ_PASSWORD=yG2g7WkufPajVUAq

echo $RADICAL_PILOT_DBURL
echo $RMQ_HOSTNAME
echo $RMQ_PORT
echo $RMQ_USERNAME
echo $RMQ_PASSWORD

export PS1="[$CONDA_PREFIX] \u@\H:\w> "
