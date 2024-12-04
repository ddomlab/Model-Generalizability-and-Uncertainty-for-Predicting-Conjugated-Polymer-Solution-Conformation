#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_out

# Correctly define models and numerical features
target_to_asses=("Rh (IW avg log)")
models_to_run=("GPR")
kernels=("matern" "rbf")

for target in "${target_to_asses[@]}"; do
    for model in "${models_to_run[@]}"; do
        for kernel in "${kernels[@]}"; do
            bsub <<EOT

#BSUB -n 8
#BSUB -W 48:01
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=32GB]"
#BSUB -J "numerical_${model}_polymer_size_feats_on_${target}_all_num"
#BSUB -o "${output_dir}/numerical_${model}_polymer_on_${target}_all_num.out"
#BSUB -e "${output_dir}/numerical_${model}_polymer_on_${target}_all_num.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_numerical_only.py --target_features "${target}" \
                                    --regressor_type "${model}" \
                                    --kernel "${kernel}" \
                                    --numerical_feats 'Mn (g/mol)' 'PDI' 'Mw (g/mol)' "Concentration (mg/ml)" "Temperature SANS/SLS/DLS/SEC (K)" "solvent dP" "solvent dD" "solvent dH" \
                                    --columns_to_impute 'PDI' 'Concentration (mg/ml)' 'Temperature SANS/SLS/DLS/SEC (K)' \
                                    --special_impute 'Mw (g/mol)' \
                                    --imputer mean

conda deactivate

EOT
        done
    done
done