#!/bin/bash

qsub -I -n 1 -t 60 -A CSC249ADCD08 -q debug-flat-quad
