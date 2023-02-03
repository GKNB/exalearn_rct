#!/bin/bash

#python serial_multiphase_rct.py \
#	--num_rank_sim 256 \
#	--num_phase 2 \
#	--config_root_dir /home/twang3/myWork/exalearn_project/run_theta/parallel_rct/test_config/

python serial_multiphase_rct.py \
	--num_rank_sim 256 \
	--num_node_ml 4 \
	--num_phase 2 \
	--num_epoch 50 \
	--config_root_dir /home/twang3/myWork/exalearn_project/run_theta/parallel_rct/test_config/ \
	--data_root_dir /lus-projects/CSC249ADCD08/twang/real_work_theta/baseline/sim/test_merge/ \
	--model_dir /home/twang3/myWork/exalearn_project/run_theta/parallel_rct/model/ \
       	--rank_in_max 64
