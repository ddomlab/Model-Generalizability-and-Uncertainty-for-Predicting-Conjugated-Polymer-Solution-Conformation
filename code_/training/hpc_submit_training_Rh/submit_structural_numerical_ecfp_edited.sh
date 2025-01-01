#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results
# Define arrays for regressor types, targets, and models
regressors=("XGBR" "NGB")
targets=("Rh (IW avg log)")
models=('ECFP')
radii=(3) 
vectors=("count")
scaler_types=('Standard' 'Robust Scaler')
poly_representations=('Monomer' 'Dimer' 'Trimer' 'RRU Monomer' 'RRU Dimer' 'RRU Trimer')

for regressor in "${regressors[@]}"; do
  for target in "${targets[@]}"; do
    for fp in "${models[@]}"; do
      for scaler in "${scaler_types[@]}"; do
        for oligo_rep in "${poly_representations[@]}"; do
          for radius in "${radii[@]}"; do
            for vector in "${vectors[@]}"; do
              bsub <<EOT



#BSUB -n 6
#BSUB -W 72:05
#BSUB -R span[ptile=2]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "${regressor}_${target}_${fp}_${scaler}_${oligo_rep}"  
#BSUB -o "${output_dir}/${regressor}_${target}_${fp}_${scaler}_${oligo_rep}_${radius}_${vector}_20250101.out"
#BSUB -e "${output_dir}/${regressor}_${target}_${fp}_${scaler}_${oligo_rep}_${radius}_${vector}_20250101.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_structure_numerical.py --target_features "${target}" \
                                      --representation "${fp}" \
                                      --regressor_type "${regressor}" \
                                      --transform_type "${scaler}" \
                                      --radius "${radius}" \
                                      --vector "${vector}" \
                                      --oligomer_representation "${oligo_rep}" \
                                      --numerical_feats 'Mn (g/mol)' 'PDI' 'Mw (g/mol)' 'Concentration (mg/ml)' 'Temperature SANS/SLS/DLS/SEC (K)' 'solvent dP' 'solvent dD' 'solvent dH' \
                                      --columns_to_impute "PDI" "Temperature SANS/SLS/DLS/SEC (K)" "Concentration (mg/ml)" \
                                      --special_impute 'Mw (g/mol)' \
                                      --imputer mean 



EOT
              done
            done
          done
        done
      done
    done
  done
done