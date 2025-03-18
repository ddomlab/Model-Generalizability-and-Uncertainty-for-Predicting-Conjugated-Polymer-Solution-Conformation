#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results/hpc_20250317
mkdir -p "$output_dir"

regressors=("XGBR" "NGB")
targets=('log Rg (nm)')
models=("Mordred" "MACCS")
poly_representations=('Monomer' 'Dimer' 'Trimer' 'RRU Monomer' 'RRU Trimer' 'RRU Dimer')


for regressor in "${regressors[@]}"; do
  for target in "${targets[@]}"; do
    for fp in "${models[@]}"; do
      for oligo_rep in "${poly_representations[@]}"; do
          bsub <<EOT
          
#BSUB -n 6
#BSUB -W 40:01
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "${fp}_${regressor}_${oligo_rep}_${target}_20250317"  
#BSUB -o "${output_dir}/${fp}_${regressor}_${oligo_rep}_${target}_20250317.out"
#BSUB -e "${output_dir}/${fp}_${regressor}_${oligo_rep}_${target}_20250317.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_structure_only.py --target_features "${target}" \
                                  --representation "${fp}" \
                                  --regressor_type "${regressor}" \
                                  --oligomer_representation "${oligo_rep}" 
EOT
      done
    done
  done
done

#'log First Peak (e-5 place holder)' 'log Second Peak (e-5 place holder)' 'log Third Peak (e-5 place holder)'
#'log First Peak wo placeholder' 'log Second Peak wo placeholder' 'log Third Peak wo placeholder'