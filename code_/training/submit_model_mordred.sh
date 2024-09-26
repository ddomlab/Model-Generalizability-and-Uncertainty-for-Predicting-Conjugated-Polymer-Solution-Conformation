#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results
bsub <<EOT

#BSUB -n 8
#BSUB -W 480
#BSUB -R span[ptile=4]
##BSUB -x
#BSUB -R "rusage[mem=32GB]"
#BSUB -J finger 
#BSUB -o ${output_dir}/mordred_run.out
#BSUB -e ${output_dir}/mordred_err.out


source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python train_structure_only.py --model mordred

conda deactivate

EOT