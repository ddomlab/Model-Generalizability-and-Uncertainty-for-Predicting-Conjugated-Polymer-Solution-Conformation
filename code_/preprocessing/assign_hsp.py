import json
from pathlib import Path
import pandas as pd
import os, sys
import numpy as np

from rdkit.Chem import Draw, MolFromSmiles, CanonSmiles, MolToSmiles
from rdkit.Chem import Mol
from rdkit.Chem import rdFingerprintGenerator
import mordred
import mordred.descriptors




# HERE: Path = Path(__file__).resolve().parent
# DATASETS: Path = HERE.parent.parent / 'datasets'
# CLEANED_DATASETS: Path = DATASETS/ 'cleaned_datasets'


# main_df_dir: Path  = CLEANED_DATASETS/'cleaned_data_wo_block_cp.csv'

# # unified_poly_name: dict[str,list] = open_json(corrected_name_dir)
# main_data = pd.read_csv(main_df_dir)
# structural_data = pd.read_pickle(structural_df_dir)

name_mod: dict = {
    'DCB':'oDCB',
    'Xyl':'mXyl',
}


def name_modify(name):
  if name in name_mod:
    name = name_mod[name]
    return name
  else:
    return name


def mixture_joining(ratios: list, solvents: list) -> str:
    ratios_str = ':'.join(ratios)
    solvents_str = '-'.join(solvents)
    result = f"{ratios_str} vol% {solvents_str}"
    return result


def mixture_spliting(sol:str)-> tuple:
      parts = sol.split(' vol% ')
      ratios = parts[0].split(':')
      solvents = parts[1].split('-')
      return ratios, solvents


def sol_name_change(entry):
  if ' vol% ' not in entry:
        entry = name_modify(entry)
        return entry
  else:
        ratio_parts,solvents =mixture_spliting(entry)

        solvents = [name_modify(sol) for sol in solvents]

        entry = mixture_joining(ratio_parts,solvents)
        return entry
  

def calculate_mixture_hsp(hsp_data,cell_value: str, hsp_param: str) -> float:
    if ' vol% ' not in cell_value:
        if cell_value == "CBd5":
            cell_value = "CB"
        pure_param = hsp_data.loc[hsp_data['Short form'] == cell_value, hsp_param]
        if pure_param.empty:
            print(f"No match found for solvent: {cell_value}")
            raise ValueError(f"No match found for solvent: {cell_value}")
        return pure_param.values[0]
    else:
        ratios, solvents = mixture_spliting(cell_value)
        if solvents[0] == "CBd5":
            solvents[0] = "CB"

        param_1 = hsp_data.loc[hsp_data['Short form'] == solvents[0], hsp_param]
        param_2 = hsp_data.loc[hsp_data['Short form'] == solvents[1], hsp_param]

        if param_1.empty:
            print(f"No match found for solvent: {solvents[0]}")
            raise ValueError(f"No match found for solvent: {solvents[0]}")
        if param_2.empty:
            print(f"No match found for solvent: {solvents[1]}")
            raise ValueError(f"No match found for solvent: {solvents[1]}")

        param_1_value = param_1.values[0]
        param_2_value = param_2.values[0]
        ratios: list[float] = [float(w) for w in ratios]
        param = np.average([param_1_value, param_2_value], weights=ratios)
        return param


def get_polymer_hsp_value(polymer_hsp_df,polymer_name, hsp_param,):
    row = polymer_hsp_df[polymer_hsp_df['Name'] == polymer_name]
    if not row.empty:
        return row[hsp_param].values[0]
    else:
        return None
    

def calculate_Ra_squared(row):
    Ra = np.sqrt(4 * (row['solvent dD'] - row['polymer dD']) ** 2 +
                  (row['solvent dP'] - row['polymer dP']) ** 2 +
                  (row['solvent dH'] - row['polymer dH']) ** 2)
    return Ra


def calculate_pairwise_hsp_distances(row, solvent_col, polymer_col):
    """
    Calculate the absolute distance between a pair of solvent and polymer values for a single row.
    """
    return np.abs(row[solvent_col] - row[polymer_col])


def map_pairwise_hsp_distances(df:pd.DataFrame):
    pairs = [
    ('solvent dD', 'polymer dD'),
    ('solvent dP', 'polymer dP'),
    ('solvent dH', 'polymer dH')
    ]
    for solvent, polymer in pairs:
        distance_column = f'abs({solvent} - {polymer})'
        df[distance_column] = df.apply(calculate_pairwise_hsp_distances, axis=1, solvent_col=solvent, polymer_col=polymer)


        
