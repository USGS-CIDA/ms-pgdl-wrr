#!/bin/bash -l
#PBS -l walltime=22:00:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N 2385496 
#PBS -o 2385496.stdout 
#PBS -q k40 

source takeme.sh

source activate pytorch4
python experiment_correlation_check.py 2385496