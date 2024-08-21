from pathlib import Path
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
RAW_dir = DATASETS/ 'raw'




def main_cleaning() -> None:
    raw_data: pd.DataFrame = pd.read_excel(RAW_dir/'Polymer_Solution_Scattering_Dataset.xlsx') 

    m_data = raw_data.applymap(lambda x: np.nan if x in ("None", 'none') else x)
    m_data=m_data.dropna(subset=['Rh1 (nm)', 'Lp (nm)', 'Rg1 (nm)'], how='all')
    print('Droping nan union of none in targets')
    print(f'shape of the dataset after droping non in target: {m_data.shape}')
    # substuting with precise value
    m_data.loc[m_data['Lp (nm)'] == ">Lc", 'Lp (nm)'] = m_data.loc[m_data['Lp (nm)'] == ">Lc", 'Lc (nm)']
    m_data.loc[m_data['Lp (nm)'] == ">>R cylinder", 'Lp (nm)'] = m_data.loc[m_data['Lp (nm)'] == ">>R cylinder", 'R core cylinder (nm)']


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

    main_columns = ["Mn (g/mol)","Mw (g/mol)","PDI", "Rh1 (nm)","Rh2 (nm)","Rh3 (nm)", "Concentration (mg/ml)", "Rg1 (nm)",
                            "Lp (nm)", "Lc (nm)", "Temperature SANS/SLS/DLS/SEC (K)"]

    for value in main_columns:
        m_data[value] = m_data[value].apply(convert_value)
    print('Done with converting to precise values ')


    for column in ["Rh1 (nm)", "Rg1 (nm)", "Lp (nm)",
                    "Lc (nm)", "Temperature SANS/SLS/DLS/SEC (K)"]:
        m_data[column] = pd.to_numeric(m_data[column], errors="coerce", )
    
    print('Done with converting to numerical values ')
    print(m_data)
    # saving cleaned dataset
    save_dir = DATASETS/'cleaned_datasets'
    m_data.to_csv(save_dir/'cleaned_pls_dataset.csv')    
    m_data.to_pickle(save_dir/'cleaned_pls_dataset.pkl')
    
main_cleaning() 

# m_data.shape
# print(m_data)
# all_poly_name = set(fp_data['name'])
# poly_smiles_name = set(structural_features['Name'])
# sym_diff2 = all_poly_name-poly_smiles_name
# (sym_diff2)


