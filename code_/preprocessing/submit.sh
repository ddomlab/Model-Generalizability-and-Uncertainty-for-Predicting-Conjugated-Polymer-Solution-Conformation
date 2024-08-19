#!/bin/bash
#BSUB -n 8
#BSUB -W 40
#BSUB -R span[ptile=4]
#BSUB -x
#BSUB -R "rusage[mem=16GB]"
#BSUB -J finger 
#BSUB -o stdout.%J
#BSUB -e stderr.%J


source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls

python /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/code_/preprocessing/fingerprint_preprocess.py

conda deactivate
