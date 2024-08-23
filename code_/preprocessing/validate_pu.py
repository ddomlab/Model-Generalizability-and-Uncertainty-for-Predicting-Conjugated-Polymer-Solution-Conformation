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

from collections import Counter

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


def check_aromatic_ring(df,row, index) -> dict:
    results: dict[str,list]  = {col: [] for col in df.columns if col != 'Monomer SMILES'}
    
    try:
        monomer_rings = count_aromatic_rings(row["Monomer SMILES"])
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

# validate RRUs

def get_atom_counts(smiles) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Counter(atom.GetSymbol() for atom in 
                       mol.GetAtoms() if atom.GetSymbol() !="*")
    else:
        raise ValueError(f"Invalid SMILES: {smiles}")
    


def check_formula(df, row, index) -> dict:
    results: dict[str,list] = {col: [] for col in df.columns if col != 'Monomer SMILES'}

    try:
        monomer_atoms = get_atom_counts(row["Monomer SMILES"])
        for col in results.keys():
            pu_atoms = get_atom_counts(row[col])
            if 'Monomer' in col and pu_atoms != monomer_atoms:
                results[col].append([index,row[col]])
            # Compare atom counts instead of ring counts
            if 'Dimer' in col and pu_atoms != {atom: 2 * count for atom, count in monomer_atoms.items()}:
                results[col].append([index, row[col]])

            if 'Trimer' in col and pu_atoms != {atom: 3 * count for atom, count in monomer_atoms.items()}:
                results[col].append([index, row[col]])

    except ValueError as e:
        print(f"Error in row {index}: {e}")

    return results


# runing
# apply used instead of iteritme() for parallelization!
def validate_formula() -> None:
    min_dir: Path = DATASETS / 'raw'
    file_path: Path = min_dir/ 'pu_processed.pkl'

    with open(file_path, 'rb') as file:
        pu_data = pd.read_pickle(file)
    
    pu_data.set_index('Name', inplace=True)
    oligomers_data: pd.DataFrame = pu_data.copy()

    invalid_smiles: pd.Series = oligomers_data.apply(lambda row: check_formula(oligomers_data,row, row.name), axis=1)

    # Consolidating the final results
    final_results: dict[str,list] = {col: [] for col in oligomers_data.columns if col != 'Monomer SMILES'}
    for res in invalid_smiles:
        for col in res:
            final_results[col].extend(res[col])
    print(final_results)


#### Applying the validation

def validate_aromaticity() -> None:
    # check number of aromatic ring in oligomers

    min_dir: Path = DATASETS / 'raw'
    file_path: Path = min_dir/ 'pu_processed.pkl'

    with open(file_path, 'rb') as file:
        pu_data = pd.read_pickle(file)
    
    oligomers_data = pu_data.set_index('Name', inplace=False)
    oligomers_data: pd.DataFrame = oligomers_data[['Monomer SMILES','Dimer SMILES', 'Trimer SMILES']]

    invalid_smiles: pd.Series = oligomers_data.apply(lambda row: check_aromatic_ring(oligomers_data,row, row.name), axis=1)
    
    # Consolidating the final results
    final_results: dict[str,list] = {col: [] for col in oligomers_data.columns if col != 'Monomer SMILES'}
    for res in invalid_smiles:
        for col in res:
            final_results[col].extend(res[col])
    print(final_results)


if __name__ == "__main__":
    validate_aromaticity()
    validate_formula()