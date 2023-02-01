# This script creates a mongodb instance. This instance will NOT be terminated when we log out, 
# which means it should only be launched once.
# The DBURL is saved in a different file, which is used in radical env setting and launching script

/grand/CSC249ADCD08/twang/libraries/mongo/bin/mongod -f /grand/CSC249ADCD08/twang/libraries/mongo/etc/mongodb.theta.conf --shutdown

/grand/CSC249ADCD08/twang/libraries/mongo/bin/mongod -f /grand/CSC249ADCD08/twang/libraries/mongo/etc/mongodb.theta.conf

export RADICAL_PILOT_DBURL="mongodb://rct:jdWeRT634k@`hostname -f`:59441/rct_db"

echo $RADICAL_PILOT_DBURL

echo $RADICAL_PILOT_DBURL > rp_dburl_polaris
