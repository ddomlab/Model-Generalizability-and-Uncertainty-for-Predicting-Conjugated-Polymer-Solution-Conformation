#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results

# Correctly define models and numerical features
target_to_assess=("Lp (nm)" "Rg1 (nm)")
models_to_run=("RF" "MLR" "DT")


for target in "${target_to_assess[@]}"; do
    for model in "${models_to_run[@]}"; do
        bsub <<EOT

#BSUB -n 8
#BSUB -W 30:01
#BSUB -R span[ptile=4]
##BSUB -x
#BSUB -R "rusage[mem=32GB]"
#BSUB -J numerical_${model}_with_feats_on_${target}
#BSUB -o ${output_dir}/numerical_${model}_polymer_${target}.out
#BSUB -e ${output_dir}/numerical_${model}_polymer_${target}.err

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python train_structure_numeric.py --target_features "${target}" \
                                  --regressor_type "${model}" \
                                  --numerical_feats 'Mn (g/mol)' 'PDI' 'Mw (g/mol)' \
                                  --columns_to_impute 'PDI' \
                                  --special_impute 'Mw (g/mol)' \
                                  --imputer mean


conda deactivate

EOT
        
    done
done