#!/bin/bash

python data_gen_helper.py \
	/home/twang3/myWork/exalearn_project/run_theta/parallel_rct/test_config/ \
	2.5 5.5 0.005 \
	2.5 5.5 0.05 \
	20 88 0.5 \
	2.5 5.5 0.05 \
	92 120 0.5 \
	3.5 4.5 0.005 \
	3.5 4.5 0.01 \
	2 \
	/home/twang3/myWork/exalearn_project/run_theta/baseline/sim/cif_file_in/ \
	/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline/sim/test_merge/

# line 4: where to generate config file
# line 13: where are cif file
# line 14: where to output hdf5 data
