module load miniconda-3
conda activate /lus-projects/CSC249ADCD08/ramki/conda-envs/rctworkflows/

export RADICAL_PILOT_DBURL=$(cat rp_dburl)

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
