#!/bin/bash


models_to_run=("RF" "MLR")
functions_to_run=("numerical" "numerical_maccs")

for model in "${models_to_run[@]}"; do
    for function in "${functions_to_run[@]}"; do
        bsub <<EOT




#BSUB -n 8
#BSUB -W 40
#BSUB -R span[ptile=4]
##BSUB -x
#BSUB -R "rusage[mem=32GB]"
#BSUB -J finger 
#BSUB -o mordred_run.out
#BSUB -e mordred_err.out


source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls

#python /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/code_/preprocessing/fingerprint_preprocess.py --num_workers 8

python train_structure_numeric.py --model $model --function $function

conda deactivate

EOT
    done
done