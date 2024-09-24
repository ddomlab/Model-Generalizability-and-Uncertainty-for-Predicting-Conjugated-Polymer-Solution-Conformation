#!/bin/bash

radii = (3 4 5 6)
vectors = ('count' 'binary')

for radius in "${radii[@]}"
do
    for vector in "${vectors[@]}"
    do
        bsub <<EOT


#BSUB -n 8
#BSUB -W 35:05
#BSUB -R span[ptile=4]
#BSUB -x
#BSUB -R "rusage[mem=32GB]"
#BSUB -J finger_radius${radius}_vector${vector}
#BSUB -o mordred_run_radius${radius}_vector${vector}.out
#BSUB -e mordred_err_radius${radius}_vector${vector}.out


source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env

#python /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/code_/preprocessing/fingerprint_preprocess.py --num_workers 8

python train_structure_only.py --radius $radius --vector $vector

conda deactivate


EOT
    done
done