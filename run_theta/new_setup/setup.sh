#/bin/bash

module load conda/2021-09-22
conda activate /lus-projects/CSC249ADCD08/twang/env/theta-work

which python
python -V
radical-stack

export RADICAL_PILOT_DBURL=$(cat /home/twang3/myWork/exalearn_project/run_theta/rp_dburl_theta)

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

echo "Need to setup libgfortran manually!"
echo "This need to be done inside RCT!!!"
export LD_LIBRARY_PATH=/home/twang3/g2full_theta/GSASII/bindist:$LD_LIBRARY_PATH

