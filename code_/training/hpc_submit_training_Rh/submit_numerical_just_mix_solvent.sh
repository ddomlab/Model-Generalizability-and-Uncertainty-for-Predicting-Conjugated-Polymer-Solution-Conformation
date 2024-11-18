#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results

# Correctly define models and numerical features
target_to_asses=("Rh (IW avg log)")
models_to_run=("RF" "MLR" "DT")


for target in "${target_to_asses[@]}"; do
    for model in "${models_to_run[@]}"; do
        bsub <<EOT

#BSUB -n 8
#BSUB -W 30:01
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=32GB]"
#BSUB -J "poly_HSP_with_${model}_on_${target}_on_polysize_hsp"
#BSUB -o "${output_dir}/poly_HSP_with_${model}_on_${target}_on_polysize_hsp.out"
#BSUB -e "${output_dir}/poly_HSP_with_${model}_on_${target}_on_polysize_hsp.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_numerical_only.py --target_features "${target}" \
                                  --regressor_type "${model}" \
                                  --numerical_feats "Mn (g/mol)" "PDI" "Mw (g/mol)" "polymer dP" "polymer dD" "polymer dH" "solvent dP" "solvent dD" "solvent dH" "Ra" \
                                  --columns_to_impute "PDI" \
                                  --special_impute 'Mw (g/mol)' \
                                  --imputer mean
                                  
conda deactivate

EOT
        
    done
done