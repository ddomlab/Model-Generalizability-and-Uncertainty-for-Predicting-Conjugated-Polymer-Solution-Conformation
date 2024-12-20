#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_out

# Correctly define models and numerical features
target_to_asses=("Rg1 (nm)")
models_to_run=("RF" "MLR" "DT")
scaler_types=('Standard' 'Robust Scaler')

for target in "${target_to_asses[@]}"; do
    for model in "${models_to_run[@]}"; do
        for scaler in "${scaler_types[@]}"; do
            bsub <<EOT

#BSUB -n 6
#BSUB -W 40:01
#BSUB -R span[ptile=2]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "numerical_${model}_polymer_size_feats_on_${target}_all_num"
#BSUB -o "${output_dir}/numerical_${model}_${scaler}_${target}_single.out"
#BSUB -e "${output_dir}/numerical_${model}_${scaler}_${target}_single.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_numerical_only.py --target_features "${target}" \
                                    --regressor_type "${model}" \
                                    --numerical_feats 'Mn (g/mol)' 'PDI' 'Mw (g/mol)' 'Concentration (mg/ml)' 'Temperature SANS/SLS/DLS/SEC (K)' "polymer dP" "polymer dD" "polymer dH" "solvent dP" "solvent dD" "solvent dH" \
                                    --imputer mean \
                                    --columns_to_impute "PDI" "Temperature SANS/SLS/DLS/SEC (K)" "Concentration (mg/ml)" \
                                    --special_impute 'Mw (g/mol)' \

conda deactivate

EOT
        done
    done
done


# "polymer dP" "polymer dD" "polymer dH" "solvent dP" "solvent dD" "solvent dH" "Ra"
# 'Mn (g/mol)' 'PDI' 'Mw (g/mol)' "Concentration (mg/ml)" "Temperature SANS/SLS/DLS/SEC (K)" "solvent dP" "solvent dD" "solvent dH"
# --special_impute 'Mw (g/mol)' \
# --columns_to_impute 'Concentration (mg/ml)' 'Temperature SANS/SLS/DLS/SEC (K)' \
# --imputer mean
