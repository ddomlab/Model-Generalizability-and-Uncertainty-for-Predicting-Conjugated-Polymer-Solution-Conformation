import pandas as pd
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
CLEANED_DATASETS = DATASETS/'cleaned_datasets'
RAW_dir = DATASETS/ 'raw'
JSONS =  DATASETS/'json_resources'

main_data = pd.read_csv(CLEANED_DATASETS/'cleaned_data_wo_block_cp.csv') 
strucutal_data = pd.read_pickle(RAW_dir/'pu_processed.pkl') 


  
all_poly_name = set(main_data['name'])
poly_unique_name = set(strucutal_data['Name'])
differences = all_poly_name-poly_unique_name

differences_json_dir = JSONS/ 'name_to_canonicalization.json'
with differences_json_dir.open('w') as file:
    json.dump(list(differences), file, indent=2)