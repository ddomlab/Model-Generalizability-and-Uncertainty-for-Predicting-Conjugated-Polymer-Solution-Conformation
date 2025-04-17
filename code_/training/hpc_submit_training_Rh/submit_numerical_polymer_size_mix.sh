#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_20250417
mkdir -p "$output_dir"

# Correctly define models and numerical features
target_to_assess=('log Rg (nm)')
models_to_run=("NGB")


for target in "${target_to_assess[@]}"; do
    for model in "${models_to_run[@]}"; do
        bsub <<EOT

#BSUB -n 4
#BSUB -W 8:01
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=8GB]"
#BSUB -J "numerical_${model}_with_feats_on_${target}_20250417"
#BSUB -o "${output_dir}/numerical_${model}_polymer_${target}_20250417.out"
#BSUB -e "${output_dir}/numerical_${model}_polymer_${target}_20250417.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_numerical_only.py --target_features "${target}" \
                                  --regressor_type "${model}" \
                                  --numerical_feats 'Xn' 'Mw (g/mol)' 'PDI' 'Concentration (mg/ml)' 'Temperature SANS/SLS/DLS/SEC (K)' "polymer dP" "polymer dD" "polymer dH" 'solvent dP' 'solvent dD' 'solvent dH'

conda deactivate

EOT
        
    done
done