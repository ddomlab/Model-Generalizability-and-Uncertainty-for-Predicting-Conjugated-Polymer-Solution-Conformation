#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_out

# Correctly define models and numerical features
target_to_asses=("Rh (IW avg log)")
models_to_run=("GPR")
kernels=("matern" "rbf")
scaler_types=('Standard' 'Robust Scaler')

for target in "${target_to_asses[@]}"; do
    for model in "${models_to_run[@]}"; do
        for kernel in "${kernels[@]}"; do
            for scaler in "${scaler_types[@]}"; do
                bsub <<EOT

#BSUB -n 4
#BSUB -W 11:01
#BSUB -R span[ptile=2]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "numerical_${model}_polymer_size_feats_on_${target}_all_num"
#BSUB -o "${output_dir}/numerical_${model}_${kernel}_${scaler}_${target}_all_num.out"
#BSUB -e "${output_dir}/numerical_${model}_${kernel}_${scaler}_${target}_all_num.err"

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
done

# "polymer dP" "polymer dD" "polymer dH" "solvent dP" "solvent dD" "solvent dH" "Ra"
# 'Mn (g/mol)' 'PDI' 'Mw (g/mol)' "Concentration (mg/ml)" "Temperature SANS/SLS/DLS/SEC (K)" "solvent dP" "solvent dD" "solvent dH"