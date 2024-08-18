#!/bin/bash
#BSUB -n 2
#BSUB -W 20
#BSUB -x
#BSUB -J 
#BSUB -o stdout.%J
#BSUB -e stderr.%J


source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls

python /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/code_/preprocessing/fingerprint_preprocess.py

conda deactivate
