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

sys.path.append("code_/cleaning")
from clean_dataset import open_json


def open_json(dir):
    with open(dir,'r') as file:
         return json.load(file)
    


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / 'datasets'
CLEANED_DATASETS: Path = DATASETS/ 'cleaned_datasets'
JSONS: Path = DATASETS/ 'json_resources'


corrected_name_dir: Path =  JSONS/'canonicalized_name.json'
main_df_dir: Path  = CLEANED_DATASETS/'cleaned_data_wo_block_cp.csv'
structural_df_dir: Path =  DATASETS/'fingerprint'/'structural_features.pkl'

unified_poly_name: dict[str,list] = open_json(corrected_name_dir)
main_data = pd.read_csv(main_df_dir)
structural_data = pd.read_pickle(structural_df_dir)



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


def map_structure():
    assign_canonical_name(main_data, 'name', unified_poly_name)
    print("Done with canonical name!")
    mapping_from_external(structural_data, main_data)
    print("Done with Mapping Structure to the main dataset!")
    print(main_data.shape)
    # saving the file 
    training_dir: Path = DATASETS/'training_dataset' 
    
    main_data.to_csv(training_dir/'structure_wo_block_cp_scaler_dataset.csv')
    main_data.to_pickle(training_dir/'structure_wo_block_cp_scaler_dataset.pkl')

map_structure()