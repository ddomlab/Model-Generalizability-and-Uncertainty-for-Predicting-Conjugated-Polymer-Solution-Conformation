# calculate the number of rings in each structure and campare it with the number of rings in the main one.
from rdkit.Chem import rdMolDescriptors
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, MolFromSmiles, CanonSmiles, MolToSmiles
from rdkit import DataStructs

from pathlib import Path
import pandas as pd
import numpy as np
import pickle

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def count_aromatic_rings(smiles: str) -> int:
    """
    Calculate the number of aromatic rings in a molecule from its SMILES string.

    Args:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        int: The number of aromatic rings.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    num_aromatic_rings = rdMolDescriptors.CalcNumRings(mol)
    return num_aromatic_rings


def validate_pu(df,row, index):
    results = {col: [] for col in df.columns if col != 'Monomer'}
    
    try:
        monomer_rings = count_aromatic_rings(row["Monomer"])
        for col in results.keys():
            pu_rings = count_aromatic_rings(row[col])
            
            # if 'Monomer' in col and pu_rings != monomer_rings:
            #     results[col].append([index,row[col]])
            
            if 'Dimer' in col and 2 * monomer_rings != pu_rings:
                results[col].append([index,row[col]])
            
            if 'Trimer' in col and 3 * monomer_rings != pu_rings:
                results[col].append([index,row[col]])

    except ValueError as e:
        print(f"Error in row {index}: {e}")
    
    return results

# Applying the validation

def validate_oligomor():
    min_dir: Path = DATASETS / 'raw'
    file_path = min_dir/ 'pu_processed.pkl'

    with open(file_path, 'rb') as file:
        pu_data = pd.read_pickle(file)
    
    pu_data.set_index('Name', inplace=True)
    oligomers_data = pu_data[['Monomer','Dimer', 'Trimer']]

    invalid_smiles = oligomers_data.apply(lambda row: validate_pu(oligomers_data,row, row.name), axis=1)

    # Consolidating the final results
    final_results = {col: [] for col in oligomers_data.columns if col != 'Monomer'}
    for res in invalid_smiles:
        for col in res:
            final_results[col].extend(res[col])
    print(final_results)



# validate_oligomor()