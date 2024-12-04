#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results
# Define arrays for regressor types, targets, and models
regressors=("GPR")
targets=("Rg1 (nm)")
radii=(3 4 5 6) 
poly_representations=('Monomer' 'Dimer' 'Trimer' 'RRU Monomer' 'RRU Dimer' 'RRU Trimer')
vectors=("count")
scaler_types=('Standard' 'Robust Scaler')
kernels=("matern" "rbf")


# Loop through each combination of regressor, target, and model
for regressor in "${regressors[@]}"; do
  for target in "${targets[@]}"; do
    for radius in "${radii[@]}"; do
      for vector in "${vectors[@]}"; do
        for oligo_rep in "${poly_representations[@]}"; do
          for scaler in "${scaler_types[@]}"; do
            for kernel in "${kernels[@]}"; do

              bsub <<EOT

#BSUB -n 8
#BSUB -W 60:01
#BSUB -R span[ptile=4]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "ecfp_${regressor}_${scaler}_${target}"  
#BSUB -o "${output_dir}/ecfp_${regressor}_${scaler}_${target}.out"
#BSUB -e "${output_dir}/ecfp_${regressor}_${scaler}_${target}.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_structure_only.py ecfp --regressor_type $regressor \
                                       --kernel "${kernel}" \
                                       --radius $radius \
                                       --vector $vector \
                                       --target "$target" \
                                       --oligo_type "$oligo_rep" \
                                       --transform_type "$scaler"
EOT
            done
          done
        done
      done
    done
  done
done