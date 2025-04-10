import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from map_structure_hsp_to_main_dataset import NumpyArrayEncoder
import json
import os, sys

sys.path.append("code_/cleaning")
from clean_dataset import open_json

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
JSONS: Path = DATASETS/ 'json_resources'

data_summary_monitor_dir = JSONS/'data_summary_monitor.json'
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.pkl"
unique_polymer_dir :Path = DATASETS/'raw'/'SMILES_to_BigSMILES_Conversion_wo_block_copolymer_with_HSPs.xlsx'

data_summary_monitor:dict[str,list] = open_json(data_summary_monitor_dir)
unique_polymer_dataset:pd.DataFrame= pd.read_excel(unique_polymer_dir)
w_data = pd.read_pickle(training_df_dir)

df_missing_poly_hsp: pd.DataFrame = w_data.copy()

# print("Size of Rh1 and 'polymer hsp' not nan:  ",
#        len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()]),
#        '   Size of Rh1 not nan\t', len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()]),
#         '   number of reduced datapoints:\t:',
#        abs(len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()])-len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()]) ))

# print("Size of Rg1 and 'polymer hsp' not nan:  ",
#        len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()]),
#        '   Size of Rg1 not nan\t', len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()]),
#        '   number of reduced datapoints:\t:',
#        abs(len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()])-len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()])) )



# print("Size of Lp and 'polymer hsp' not nan:  ",
#        len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()]),
#        '   Size of Lp not nan\t', len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()]),
#        '   number of reduced datapoints:\t:',
#        abs(len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()])-len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()]))) 

# reduced_df = df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull() & df_missing_poly_hsp['polymer dH'].isnull()]

# # Show unique polymer names/types that are reduced
# reduced_polymer_types = reduced_df['canonical_name'].unique()

# print(f"Reduced polymer types:{reduced_polymer_types}")



# print("full size of the dataset without additives:  ", len(df_missing_poly_hsp))

# print("full size of the dataset without additives and no nan in hsp:  ", len(df_missing_poly_hsp[~df_missing_poly_hsp['polymer dH'].isnull()]))

# print("Number of unique polymers with missing hsp in training datsaet" ,df_missing_poly_hsp['polymer dH'].isna().sum())


targets = ['Rh1 (nm)', 'Rg1 (nm)', 'Lp (nm)']

if __name__ == "__main__":
    df_missing_poly_hsp_unique_poly = unique_polymer_dataset[['Name','SMILES','dD', 'dP','dH']].copy()

    df_without_hsp = df_missing_poly_hsp_unique_poly[df_missing_poly_hsp_unique_poly['dD'].isna()].reset_index(drop=True)
    df_without_hsp.to_csv(DATASETS/'raw'/'polymer_without_hsp.csv',index=False)
    print("Unique polymers without HSP values saved to polymer_without_hsp.csv!")



    df_training_dropped_missing_polymer_hsp = w_data.dropna(subset=["polymer dH"]).reset_index(drop=True)
    after_dropping_polymer_hsps_counts = {t: df_training_dropped_missing_polymer_hsp[t].notna().sum() for t in targets}

    print("Drop missing HSP values for polymers")
    print(df_training_dropped_missing_polymer_hsp['Rh (IW avg log)'])
    df_training_dropped_missing_polymer_hsp.to_pickle(DATASETS/"training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl")
    print("Done saving dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl")

    Rh_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Rh (IW avg log)'].notna()]
    Rg_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Rg1 (nm)'].notna()]
    Lp_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Lp (nm)'].notna()]


    for t in targets:
        data_summary_monitor['After dropping unknown polymer HSPs'].append(after_dropping_polymer_hsps_counts[t])
    with open(JSONS/"data_summary_monitor.json", "w") as f:
        json.dump(data_summary_monitor, f, cls=NumpyArrayEncoder, indent=2)
