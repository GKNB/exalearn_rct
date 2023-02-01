#!/bin/bash

qsub -I -l select=1,walltime=01:00:00 -l filesystems=home:grand:eagle -A CSC249ADCD08 -q debug
