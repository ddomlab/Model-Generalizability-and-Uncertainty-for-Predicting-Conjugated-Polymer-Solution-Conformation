#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_20250204
# Define arrays for regressor types, targets, and models
mkdir -p "$output_dir"
regressors=("XGBR")
target_to_asses=('log First Peak wo placeholder' 'log Second Peak wo placeholder' 'log Third Peak wo placeholder')
models=("Mordred" "MACCS")
poly_representations=('Dimer' 'RRU Dimer')
# scaler_types=("Robust Scaler")
# kernels=("matern" "rbf")

# Loop through each combination of regressor, target, and model
for regressor in "${regressors[@]}"; do
  for target in "${target_to_asses[@]}"; do
    for model in "${models[@]}"; do
      for oligo_rep in "${poly_representations[@]}"; do
        # for scaler in "${scaler_types[@]}"; do
          # for kernel in "${kernels[@]}"; do
              bsub <<EOT
          
#BSUB -n 6
#BSUB -W 40:01
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "mordred_${regressor}_${scaler}_${target}_20250204"  
#BSUB -o "${output_dir}/${model}_${regressor}_${oligo_rep}_${target}_20250204.out"
#BSUB -e "${output_dir}/${model}_${regressor}_${oligo_rep}_${target}_20250204.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_structure_only.py --target_features "${target}" \
                                  --representation "${fp}" \
                                  --regressor_type "${regressor}" \
                                  --oligomer_representation "${oligo_rep}" \
EOT
        # done
      done
    done
  done
done

# --kernel "${kernel}" \
