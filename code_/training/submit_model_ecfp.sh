#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results

radii = (3 4 5 6)
vectors = ('count' 'binary')

for radius in "${radii[@]}"; do
  for vector in "${vectors[@]}"; do
    bsub <<EOT
#BSUB -n 8
#BSUB -W 35:05
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=32GB]"
#BSUB -J ${output_dir}/finger_radius${radius}_vector${vector}
#BSUB -o ${output_dir}/ecfp_run_radius${radius}_vector${vector}.out
#BSUB -e ${output_dir}/ecfp_err_radius${radius}_vector${vector}.out

source ~/.bashrc
conda activate /share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/env-pls
python train_structure_only.py --model ecfp --radius ${radius} --vector ${vector}
EOT
  done
done
