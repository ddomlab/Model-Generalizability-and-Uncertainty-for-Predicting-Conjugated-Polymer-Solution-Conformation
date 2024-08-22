import json
from pathlib import Path
import pandas as pd
import os, sys
sys.path.append("code_/cleaning")
from clean_dataset import open_json

print(f'sys.path {sys.path}')

# HERE: Path = Path(__file__).resolve().parent
# DATASETS: Path = HERE.parent.parent / 'datasets'
# JSONS: Path = DATASETS/ 'json_resources'


# corrected_name_dir = JSONS/'canonicalized_name.json'
# unified_poly_name = open_json(corrected_name_dir)

# def mapping_from_external(source, to_main):
#         working_main= to_main.copy()
#         working_structure = source.set_index('Name', inplace=False)

#         # Map combined tuples to the main dataset
#         combined_series = working_structure.apply(lambda row: tuple(row.values), axis=1)
#         mapped_data = working_main['canonical_name'].map(combined_series)
#         unpacked_data = list(zip(*mapped_data))
#         print('yes')
#         # Assign the unpacked data to the corresponding columns in the dataset
#         for idx, col in enumerate(working_structure.columns.tolist()):
#             working_main[col] = unpacked_data[idx]
#         return working_main


# print(unified_poly_name)
