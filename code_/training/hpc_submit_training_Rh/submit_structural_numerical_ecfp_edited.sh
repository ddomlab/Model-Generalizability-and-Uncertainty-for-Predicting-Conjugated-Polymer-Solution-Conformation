#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_20250205
mkdir -p "$output_dir"

regressors=("XGBR")
targets=('Rh (1_1000 nm) (highest intensity)')
models=("ECFP")
radii=(3) 
vectors=("count")
poly_representations=('Dimer' 'RRU Dimer' 'Monomer' 'RRU Monomer' 'Trimer' 'RRU Trimer')

for regressor in "${regressors[@]}"; do
  for target in "${targets[@]}"; do
    for fp in "${models[@]}"; do
      for oligo_rep in "${poly_representations[@]}"; do
        for radius in "${radii[@]}"; do
          for vector in "${vectors[@]}"; do
              bsub <<EOT



#BSUB -n 8
#BSUB -W 72:05
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "${regressor}_${target}_${fp}_${oligo_rep}_20250205"  
#BSUB -o "${output_dir}/${regressor}_${target}_${fp}_${oligo_rep}_${radius}_${vector}_20250205.out"
#BSUB -e "${output_dir}/${regressor}_${target}_${fp}_${oligo_rep}_${radius}_${vector}_20250205.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_structure_numerical.py --target_features "${target}" \
                                      --representation "${fp}" \
                                      --regressor_type "${regressor}" \
                                      --radius "${radius}" \
                                      --vector "${vector}" \
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
  done
done

# 'Monomer' 'Dimer' 'Trimer' 'RRU Monomer' 'RRU Dimer'