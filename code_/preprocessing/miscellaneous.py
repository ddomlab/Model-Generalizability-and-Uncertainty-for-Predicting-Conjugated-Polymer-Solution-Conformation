import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from visualization.visualization_setting import save_img_path


HERE: Path = Path(__file__).resolve().parent
VISUALIZATION = HERE.parent/ "visualization"
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "Rg data with clusters.pkl"
w_data = pd.read_pickle(training_df_dir)
w_data = w_data.reset_index(drop=True)


def calculate_dp(smiles, polymer_mw):
    """Calculate the degree of polymerization (DP) for a given monomer SMILES and polymer molecular weight."""
    try:
        monomer = Chem.MolFromSmiles(smiles)
        if monomer is None:
            raise ValueError("Invalid SMILES string")
        
        monomer_mw = Descriptors.MolWt(monomer) 
        if monomer_mw == 0:
            raise ValueError("Monomer molecular weight is zero. Check the SMILES string.")
        
        return polymer_mw / monomer_mw  
    
    except Exception as e:
        return str(e)  



w_data['log Rg (nm)'] = np.log10(w_data['Rg1 (nm)'])
w_data['DP'] = w_data.apply(
    lambda row: calculate_dp(row['Monomer SMILES'], row['Mw (g/mol)']), axis=1
)
w_data.to_pickle(training_df_dir)

