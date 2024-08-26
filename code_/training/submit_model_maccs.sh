#!/bin/bash
#BSUB -n 8
#BSUB -W 40
#BSUB -R span[ptile=4]
##BSUB -x
#BSUB -R "rusage[mem=32GB]"
#BSUB -J finger 
#BSUB -o maccs_run.out
#BSUB -e maccs_err.out


source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls

#python /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/code_/preprocessing/fingerprint_preprocess.py --num_workers 8

python train_structure_only.py --model maccs

conda deactivate
