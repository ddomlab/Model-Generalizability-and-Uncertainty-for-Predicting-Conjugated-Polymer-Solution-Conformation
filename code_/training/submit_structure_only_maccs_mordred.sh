#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results
# Define arrays for regressor types, targets, and models
regressors=("NGB")
targets=("Rg1 (nm)")
models=("maccs" "mordred")
poly_representations=('Monomer' 'Dimer' 'Trimer' 'RRU Monomer' 'RRU Dimer' 'RRU Trimer')




# Loop through each combination of regressor, target, and model
for regressor in "${regressors[@]}"; do
  for target in "${targets[@]}"; do
    for model in "${models[@]}"; do
      for oligo_rep in ${poly_representations[@]}; do
      
      bsub <<EOT
#BSUB -n 8
#BSUB -W 70:05
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=32GB]"
#BSUB -J "${oligo_rep}_${target}_${model}_${regressor}_structure_only"  
#BSUB -o ${output_dir}/model_${model}_${target}_${regressor}_${oligo_rep}_structure_only.out
#BSUB -e ${output_dir}/model_${model}_${target}_${regressor}_${oligo_rep}_structure_only.err

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python train_structure_only.py $model --regressor_type $regressor --target "$target" --oligo_type $oligo_rep
EOT
      done
    done
  done
done