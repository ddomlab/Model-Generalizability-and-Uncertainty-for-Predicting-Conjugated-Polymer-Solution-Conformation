#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_20250205
mkdir -p "$output_dir"

# Define arrays for regressor types, targets, and models
regressors=("XGBR")
targets=("target_Rh (1_1000 nm) (highest intensity)_LogFT")
models=("Mordred" "MACCS")
poly_representations=('Dimer' 'RRU Dimer')

for regressor in "${regressors[@]}"; do
  for target in "${targets[@]}"; do
    for fp in "${models[@]}"; do
      for oligo_rep in "${poly_representations[@]}"; do
          bsub <<EOT



#BSUB -n 6
#BSUB -W 50:05
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "${regressor}_${target}_${fp}_${oligo_rep}_20250221"  
#BSUB -o "${output_dir}/${regressor}_${target}_${fp}_${oligo_rep}_20250221.out"
#BSUB -e "${output_dir}/${regressor}_${target}_${fp}_${scaler}_${oligo_rep}_20250221.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_structure_numerical.py --target_features "${target}" \
                                      --representation "${fp}" \
                                      --regressor_type "${regressor}" \
                                      --oligomer_representation "${oligo_rep}" \
                                      --numerical_feats 'Mn (g/mol)' 'PDI' 'Mw (g/mol)' 'Concentration (mg/ml)' 'Temperature SANS/SLS/DLS/SEC (K)' "polymer dP" "polymer dD" "polymer dH" 'solvent dP' 'solvent dD' 'solvent dH' \
                                      --columns_to_impute "PDI" "Temperature SANS/SLS/DLS/SEC (K)" "Concentration (mg/ml)" \
                                      --special_impute 'Mw (g/mol)' \
                                      --imputer mean 



EOT
      done
    done
  done
done