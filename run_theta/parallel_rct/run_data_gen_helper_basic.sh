#!/bin/bash

exp_dir_name=$1
exp_idx=$2

python ./data_gen_helper.py \
	${exp_dir_name}/Ex${exp_idx}/configs/ \
	2.5 5.5 0.001 \
	2.5 5.5 0.01 \
	20 88 0.5 \
	2.5 5.5 0.01 \
	92 120 0.5 \
	3.5 4.5 0.005 \
	3.5 4.5 0.002 \
	1 \
	/lus-projects/CSC249ADCD08/twang/real_work_theta/code/cif_file_in/ \
	${exp_dir_name}/Ex${exp_idx}/data

# line 4: where to generate config file
# line 13: where are cif file
# line 14: where to output hdf5 data
