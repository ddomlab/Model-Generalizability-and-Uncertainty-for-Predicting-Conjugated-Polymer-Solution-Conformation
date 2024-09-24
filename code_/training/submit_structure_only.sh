#!/bin/bash

# Define radii and vector types
radii=(3 4 5 6)
vectors=("count" "binary")

# Submit perform_model_mordred as a job
bsub <<EOT
#BSUB -n 8
#BSUB -W 35:05
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=32GB]"
#BSUB -J mordred_model
#BSUB -o mordred_run.out
#BSUB -e mordred_err.out

source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls
python train_structure_only.py --model mordred
EOT

# Submit perform_model_maccs as a job
bsub <<EOT
#BSUB -n 8
#BSUB -W 35:05
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=32GB]"
#BSUB -J maccs_model
#BSUB -o maccs_run.out
#BSUB -e maccs_err.out

source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls
python train_structure_only.py --model maccs
EOT

# Submit perform_model_ecfp for each radius and vector combination as separate jobs
for radius in "${radii[@]}"; do
  for vector in "${vectors[@]}"; do
    bsub <<EOT
#BSUB -n 8
#BSUB -W 35:05
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=32GB]"
#BSUB -J finger_radius${radius}_vector${vector}
#BSUB -o ecfp_run_radius${radius}_vector${vector}.out
#BSUB -e ecfp_err_radius${radius}_vector${vector}.out

source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls
python train_structure_only.py --model ecfp --radius ${radius} --vector ${vector}
EOT
  done
done
