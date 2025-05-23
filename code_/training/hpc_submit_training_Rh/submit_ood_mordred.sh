#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_20250515
mkdir -p "$output_dir"

regressors=("MLP")
targets=("log Rg (nm)")
models=("Mordred")
poly_representations=("Trimer")
group_out=('substructure cluster')
# 'KM3 Mordred cluster' 'substructure cluster' 'KM4 polymer_solvent HSP and polysize cluster' 'KM5 polymer_solvent HSP and polysize cluster' 'KM4 polymer_solvent HSP cluster' 'KM4 Mordred_Polysize cluster'
# 'KM3 Mordred cluster'
# 'KM4 Mordred_Polysize cluster'

for regressor in "${regressors[@]}"; do
  for target in "${targets[@]}"; do
    for fp in "${models[@]}"; do
      for oligo_rep in "${poly_representations[@]}"; do
          for group in "${group_out[@]}"; do
            bsub <<EOT



#BSUB -n 6
#BSUB -W 2:30
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "${regressor}_${target}_${fp}_${oligo_rep}_${group}_20250515"  
#BSUB -o "${output_dir}/${regressor}_${target}_${fp}_${oligo_rep}_${group}_20250515.out"
#BSUB -e "${output_dir}/${regressor}_${target}_${fp}_${oligo_rep}_${group}_20250515.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../make_ood_prediction.py --target_features "${target}" \
                                  --representation "${fp}" \
                                  --regressor_type "${regressor}" \
                                  --oligomer_representation "${oligo_rep}" \
                                  --numerical_feats 'Xn' 'Mw (g/mol)' 'PDI' 'Concentration (mg/ml)' 'Temperature SANS/SLS/DLS/SEC (K)' "polymer dP" "polymer dD" "polymer dH" 'solvent dP' 'solvent dD' 'solvent dH' \
                                  --clustering_method "${group}" \



EOT
        done
      done
    done
  done
done

# 'Monomer' 'Dimer' 'Trimer' 'RRU Monomer' 'RRU Dimer'