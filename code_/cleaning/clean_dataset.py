import pandas as pd
import numpy as np
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
RAW_dir = DATASETS/ 'raw'
JSONS =  DATASETS/'json_resources'

def open_json(dir):
    with open(dir,'r') as file:
         return json.load(file)

def convert_value(value):
    if isinstance(value, (int, float)):
        return value
    elif '-' in str(value) or '−' in str(value):  # Added handling for en dash character
        parts = str(value).replace('−', '-').split('-')  # Replacing en dash with hyphen
        if len(parts) == 2:  # If it's in the format number1-number2
            num1, num2 = map(float, parts)
            return (num1 + num2) / 2
        # elif len(parts) == 1:
        #     return sum(map(float, parts)) / len(parts)

    elif '/' in str(value):
        parts = str(value).split('/')
        return sum(map(float, parts)) / len(parts)
    elif str(value).startswith('>'):
        return float(value[1:]) + 1
    elif str(value).startswith('<'):
        return float(value[1:]) - 1
    else:
        return float(value)
def main_cleaning(raw_data:pd.DataFrame) -> None:

    m_data = raw_data.applymap(lambda x: np.nan if x in ("None", 'none') else x)
    m_data=m_data.dropna(subset=['Rh1 (nm)', 'Lp (nm)', 'Rg1 (nm)'], how='all')
    print('Droping nan union of none in targets')
    print(f'shape of the dataset after droping non in target: {m_data.shape}')
    # substuting with precise value
    m_data.loc[m_data['Lp (nm)'] == ">Lc", 'Lp (nm)'] = m_data.loc[m_data['Lp (nm)'] == ">Lc", 'Lc (nm)']
    m_data.loc[m_data['Lp (nm)'] == ">>R cylinder", 'Lp (nm)'] = m_data.loc[m_data['Lp (nm)'] == ">>R cylinder", 'R core cylinder (nm)']

    main_columns = ["Mn (g/mol)","Mw (g/mol)","PDI", "Rh1 (nm)","Rh2 (nm)","Rh3 (nm)", "Concentration (mg/ml)", "Rg1 (nm)",
                            "Lp (nm)", "Lc (nm)", "Temperature SANS/SLS/DLS/SEC (K)"]

    for value in main_columns:
        m_data[value] = m_data[value].apply(convert_value)
    print('Done with converting to precise values ')

    numreical_feats: list[str] = ["Rh1 (nm)", "Rg1 (nm)", "Lp (nm)",
                    "Lc (nm)", "Temperature SANS/SLS/DLS/SEC (K)"]
    for column in numreical_feats:
        m_data[column] = pd.to_numeric(m_data[column], errors="coerce", )
    
    m_data = m_data.reset_index(drop=True)
    print('Done with converting to numerical values ')
    print(f'm_data shape: {m_data.shape}')
    # saving cleaned dataset
    save_dir = DATASETS/'cleaned_datasets'
    m_data.to_csv(save_dir/'cleaned_pls_dataset.csv')    
    m_data.to_pickle(save_dir/'cleaned_pls_dataset.pkl')
    return m_data


    
def drop_block_cp(m_data:pd.DataFrame):
    block_cp_dir = JSONS/'block_copolymers.json'
    block_cp : dict[str,list] = open_json(block_cp_dir) 
    values_to_drop: list[str] = [poly_name for poly_name in block_cp.keys()]

    m_data: pd.DataFrame = m_data[~m_data['name'].isin(values_to_drop)].reset_index(drop=True)

    save_dir = DATASETS/'cleaned_datasets'
    m_data.to_csv(save_dir/'cleaned_data_wo_block_cp.csv')
    m_data.to_pickle(save_dir/'cleaned_data_wo_block_cp.pkl')
    print('Done with dropping block copolymers')
    print(f'block_cp dataset shape: {m_data.shape}')





