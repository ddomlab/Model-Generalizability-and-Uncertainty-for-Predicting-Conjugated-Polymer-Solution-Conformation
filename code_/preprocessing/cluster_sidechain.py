import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem

HERE: Path = Path(__file__).resolve().parent
VISUALIZATION = HERE.parent/ "visualization"
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal (40-1000 nm)_added.pkl"
w_data = pd.read_pickle(training_df_dir)




alkyl_smarts_patterns = []
for i in range(2, 13):  # From C2 (ethyl) to C12 (dodecyl)
    alkyl_smarts_patterns.append('[CX4]' * i)  # Alkyl chain with i carbons

# polar_smarts = Chem.MolFromSmarts('[OH]')  # Hydroxyl group
ether_smarts = Chem.MolFromSmarts('[OX2;!r]')
ester_smarts = Chem.MolFromSmarts('[C;!r](=O)[O;!r][C;!r]')
ionic_smarts = Chem.MolFromSmarts('[+,-]')  # Matches charged atoms (ionic)
alkyl_cyanide = Chem.MolFromSmarts('[N;!r]CC') 

def determine_side_chain_type(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if not mol:
            return "Invalid SMILES"

        if mol.HasSubstructMatch(ionic_smarts):
            return "ionic"

        elif mol.HasSubstructMatch(ether_smarts) or mol.HasSubstructMatch(ester_smarts):
            return "polar"

        if mol.HasSubstructMatch(alkyl_cyanide):
            return 'polar'

        for alkyl_smarts in alkyl_smarts_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(alkyl_smarts)):
                return "non-polar"
        


        return "other"
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return "Error"
    

if __name__ == "__main__":
    Rh_data = w_data[w_data["multimodal Rh"].notna()]
    Rh_data["Side Chain type"] = Rh_data["Monomer SMILES"].apply(determine_side_chain_type)

    Rg_data = w_data[w_data['Rg1 (nm)'].notna()]
    Rg_data["Side Chain type"] = Rg_data["Monomer SMILES"].apply(determine_side_chain_type)

    unique_data = w_data[['canonical_name', 'Monomer SMILES']].drop_duplicates()
    unique_data["Side Chain type"] = unique_data["Monomer SMILES"].apply(determine_side_chain_type)
    w_data["Side Chain type"] = w_data["Monomer SMILES"].apply(determine_side_chain_type)
    
    for group in ['non-polar', 'polar', 'ionic']:
        # print(f"{group}:",len(w_data[w_data['Side Chain type']==group]))
        # print(f"unique {group}:",len(unique_data[unique_data['Side Chain type']==group]))
        print(f"Rg: {group}:",len(Rg_data[Rg_data['Side Chain type']==group]))
        print(f"Rh: {group}:",len(Rh_data[Rh_data['Side Chain type']==group]))