<img width="1963" height="2020" alt="image" src="https://github.com/user-attachments/assets/1124a05f-6a82-4e87-ad7f-83604d6f76e9" /><br><br>

This repository contains the data, preprocessing workflows, and machine learning pipelines developed for the paper:<br>
"**Robust Learning from Literature Data: Model Generalizability and Uncertainty for Predicting Conjugated Polymer Solution Conformation**"


## Overview
The repository is set-up to make the results easy to reproduce. If you get stuck or like to learn more, please feel free to open an issue.

## Setup
The environment.yml file specifies the conda virtual environment. :<br>
<pre> conda env create -f environment.yml </pre>

##  Repository Structure

```bash
code_/                       
├── cleaning/                
│   └── generate_clean_dataset.py             # Main script to clean and prepare dataset
├── preprocessing/           
│   ├── handle_pu.py                          # Handles oligomers and polymer repeat units and
│   ├── map_structure_hsp_to_main_dataset.py  # Maps molecular representations and HSPs to dataset
│   └── drop_unknown_hsps.py                  # Drops entries with missing/unknown HSP values
├── training/                
│   ├── all_factories.py                      # All necessary functions and operators 
│   ├── get_ood_split.py                      # Define OOD train/test splits
│   ├── get_ood_split_learning_curve.py       # OOD learning curve experiment
│   ├── imputation_normalization.py           # Imputation and normalization function
│   ├── learning_curve_utils.py               # Shared utilities for learning curves
│   ├── make_ood_learning_curve.py            # Make OOD learning curve results
│   ├── make_ood_prediction.py                # Make OOD predictions
│   ├── scoring.py                            # Evaluation metrics and cross validations
│   ├── train_structure_numerical_generalized.py  # Random seeds for reproducibility
│   ├── train_structure_numerical.py          # Train with both structural or/and numerical
│   ├── training_utils.py                     # Shared training helpers
│   ├── unrolling_utils.py                    # Unrolling utilities for molecular representations
├── visualization/           
│   ├── visualization_setting.py              # Plot style/setting configs
│   ├── visualize_heatmap.py                  # Heatmap plotting
│   ├── visualize_IID_learning_curve.py       # Visualize IID learning curves
│   ├── visualize_ood_full_data.py            # Visualize full OOD dataset results
│   ├── visualize_ood_learning_curve.py       # Visualize OOD learning curves
│   ├── visualize_predictions_truth.py        # Prediction vs truth Hex plots
│   └── visualize_uncertainty_calibration.py  # Calibration plots for uncertainty
├── datasets/                    
│   ├── fingerprint/
|       ├── structural_features.csv           # Molecular representation for mapping to dataset   
│   ├── json_resources/
|       ├── block_copolymers.json                       # name of block copolymer to remove 
|       ├── canonicalized_name.json                     # Canonicalized polymer naming references
|       ├── data_summary_monitor.json                   # Dataset cleaning and summary tracking
|       └── name_to_canonicalization.json               # Name → canonical form lookup table 
│   ├── raw/                                  # Raw curated datasets
│       ├── Polymer_Solution_Scattering_Dataset.xlsx   # Initial collected data
│       ├── polymer_without_hsp.csv                    # Dataset excluding Hansen solubility parameters
│       ├── pu_processed.csv                           # Processed polymer repeat units and oligomers (CSV)
│       └── SMILES_to_BigSMILES_Conversion_wo_block_copolymer_with_HSPs.xlsx  # List of SMILES and HSPs of polymers
│                      
└── training_dataset/
    ├── Rg data with clusters aging imputed.pkl   # Final cleaned dataset including imputed aging parameters and clusters for OOD evaluation

results/                                   
├── HPC history                            # Logs and history of HPC job submissions/runs
├── OOD_target_log Rg (nm)                 # Out-of-distribution prediction results for log Rg
└── target_log Rg (nm)                     # In-distribution prediction results for log Rg         

env-pls-dataset.yml          
LICENSE
README.md
```
## How to cite 
