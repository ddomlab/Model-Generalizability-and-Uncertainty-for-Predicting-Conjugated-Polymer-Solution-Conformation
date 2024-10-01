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




HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / 'datasets'
CLEANED_DATASETS: Path = DATASETS/ 'cleaned_datasets'


main_df_dir: Path  = CLEANED_DATASETS/'cleaned_data_wo_block_cp.csv'

# unified_poly_name: dict[str,list] = open_json(corrected_name_dir)
main_data = pd.read_csv(main_df_dir)
structural_data = pd.read_pickle(structural_df_dir)
