#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_out

# Correctly define models and numerical features
target_to_asses=("Rh (IW avg log)")
models_to_run=("RF" "MLR" "DT")
numerical_feats=("polymer dP" "polymer dD" "polymer dH" "solvent dP" "solvent dD" "solvent dH" "Ra")

for target in "${target_to_asses[@]}"; do
        for model in "${models_to_run[@]}"; do
            for feats in "${numerical_feats[@]}"; do
                
                bsub <<EOT

#BSUB -n 8
#BSUB -W 10:01
#BSUB -R span[ptile=4]
##BSUB -x
#BSUB -R "rusage[mem=32GB]"
#BSUB -J "numerical_${model}_with_${feats}_on_${target}"
#BSUB -o "${output_dir}/numerical_${model}_with_${feats}_on_${target}.out"
#BSUB -e "${output_dir}/numerical_${model}_with_${feats}_on_${target}.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env

python ../train_numerical_only.py --target_features "${target}" \
                                  --regressor_type "${model}" \
                                  --numerical_feats "${feats}" \

conda deactivate

EOT
        done
    done
done


