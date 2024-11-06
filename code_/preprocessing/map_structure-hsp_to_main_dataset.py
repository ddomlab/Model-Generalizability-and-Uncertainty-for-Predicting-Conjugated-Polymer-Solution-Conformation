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

from assign_hsp import (sol_name_change, 
                        calculate_mixture_hsp,
                        get_polymer_hsp_value,
                        calculate_Ra_squared)


sys.path.append("code_/cleaning")
from clean_dataset import open_json


# def open_json(dir):
#     with open(dir,'r') as file:
#          return json.load(file)
    

# file paths

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / 'datasets'
CLEANED_DATASETS: Path = DATASETS/ 'cleaned_datasets'
JSONS: Path = DATASETS/ 'json_resources'
RAW_dir = DATASETS/ 'raw'


corrected_name_dir: Path =  JSONS/'canonicalized_name.json'
main_df_dir: Path  = CLEANED_DATASETS/'cleaned_data_wo_block_cp.pkl'
structural_df_dir: Path =  DATASETS/'fingerprint'/'structural_features.pkl' # fingerprints data in array format

# reading files
unified_poly_name: dict[str,list] = open_json(corrected_name_dir)
main_data = pd.read_pickle(main_df_dir)
structural_data = pd.read_pickle(structural_df_dir)

# reading hsp features
raw_solvent_properties: pd.DataFrame = pd.read_excel(RAW_dir/'Polymer_Solution_Scattering_Dataset.xlsx',
                                                     sheet_name='solvents_addtives_short_form') 
raw_polymer_hsp: pd.DataFrame = pd.read_excel(RAW_dir/'SMILES_to_BigSMILES_Conversion_wo_block_copolymer_with_HSPs.xlsx')






def assign_canonical_name(df, original_name: str, mapping: dict) -> pd.DataFrame:
    """
    Assigns canonical name to the dataset.
    """
    canon_poly_name = {v: k for k, values in mapping.items() for v in values}

    def map_to_canonical_nam(name):
        return canon_poly_name.get(name, name)

    df['canonical_name'] = df[original_name].apply(map_to_canonical_nam)

def mapping_from_external(strucure_df, main_df):
    
        working_structure = strucure_df.set_index('Name', inplace=False)
        # Map combined tuples to the main dataset
        combined_series = working_structure.apply(lambda row: tuple(row.values), axis=1)
        mapped_data = main_df['canonical_name'].map(combined_series)
        unpacked_data = list(zip(*mapped_data))
        # Assign the unpacked data to the corresponding columns in the dataset
        for idx, col in enumerate(working_structure.columns.tolist()):
            main_df[col] = unpacked_data[idx]
        # map the hsp of polymer here


def map_structure():
    assign_canonical_name(main_data, 'name', unified_poly_name)
    print("Done with canonical name!")
    mapping_from_external(structural_data, main_data)
    print("Done with Mapping Structure to the main dataset!")
    print(main_data.shape)
    # saving the file 
    return main_data



# Apply the function to each row
# df['Ra_squared'] = df.apply(calculate_Ra_squared, axis=1)

def map_hsp(df,solvent_df,polymer_hsp):
    df['modified_solvent_format'] = df['Solvent(s)'].apply(sol_name_change)
    
    hsp_param_names:list[str] = ['dP', 'dD', 'dH']
    for param in hsp_param_names:
        df[f'solvent {param}'] = df['modified_solvent_format'].apply(lambda x: calculate_mixture_hsp(solvent_df,x, param))
        df[f'polymer {param}'] = df['canonical_name'].apply(lambda x: get_polymer_hsp_value(polymer_hsp,x, param))
    print('Done with mapping solvent and polymer hsp')
    
    df['Ra'] = df.apply(calculate_Ra_squared, axis=1)
    print('Done with calculating Ra')

    return df


def generate_training_dataset():
    training_dir: Path = DATASETS/'training_dataset' 

    dataset_structure_added: pd.DataFrame= map_structure()
    dataset_structure_added.to_csv(training_dir/'dataset_wo_block_cp_fp_added.csv',index=False)
    dataset_structure_added.to_pickle(training_dir/'dataset_wo_block_cp_fp_added.pkl')
    dataset_hsp_added: pd.DataFrame = map_hsp(dataset_structure_added,raw_solvent_properties,raw_polymer_hsp)
    dataset_hsp_added.to_csv(training_dir/'dataset_wo_block_cp_(fp-hsp)_added.csv',index=False)
    dataset_hsp_added.to_pickle(training_dir/'dataset_wo_block_cp_(fp-hsp)_added.pkl')
    dataset_hsp_added_dropped_additives = dataset_hsp_added[dataset_hsp_added['Solid additive'].isna()].reset_index(drop=True) 
    dataset_hsp_added_dropped_additives.to_csv(training_dir/'dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.csv',index=False)
    dataset_hsp_added_dropped_additives.to_pickle(training_dir/'dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.pkl')


if __name__ == "__main__":
    generate_training_dataset()
