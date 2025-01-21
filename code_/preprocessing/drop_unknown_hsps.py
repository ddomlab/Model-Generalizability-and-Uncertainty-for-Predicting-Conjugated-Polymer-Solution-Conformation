import pandas as pd
from pathlib import Path
import numpy as np



HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.pkl"
unique_polymer_dir :Path = DATASETS/'raw'/'SMILES_to_BigSMILES_Conversion_wo_block_copolymer_with_HSPs.xlsx'
unique_polymer_dataset:pd.DataFrame= pd.read_excel(unique_polymer_dir)




w_data = pd.read_pickle(training_df_dir)

df_missing_poly_hsp: pd.DataFrame = w_data[['Rh1 (nm)', 'Rg1 (nm)','Lp (nm)','polymer dH']].copy()

print("Size of Rh1 and 'polymer hsp' not nan:  ",
       len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()]),
       '   Size of Rh1 not nan\t', len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()]),
        '   number of reduced datapoints:\t:',
       abs(len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()])-len(df_missing_poly_hsp[~df_missing_poly_hsp["Rh1 (nm)"].isnull()]) ))

print("Size of Rg1 and 'polymer hsp' not nan:  ",
       len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()]),
       '   Size of Rg1 not nan\t', len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()]),
       '   number of reduced datapoints:\t:',
       abs(len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()])-len(df_missing_poly_hsp[~df_missing_poly_hsp['Rg1 (nm)'].isnull()])) )

print("Size of Lp and 'polymer hsp' not nan:  ",
       len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()]),
       '   Size of Lp not nan\t', len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()]),
       '   number of reduced datapoints:\t:',
       abs(len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()&~df_missing_poly_hsp['polymer dH'].isnull()])-len(df_missing_poly_hsp[~df_missing_poly_hsp['Lp (nm)'].isnull()]))) 

print("full size of the dataset without additives:  ", len(df_missing_poly_hsp))

print("full size of the dataset without additives and no nan in hsp:  ", len(df_missing_poly_hsp[~df_missing_poly_hsp['polymer dH'].isnull()]))

print("Number of unique polymers with missing hsp in training datsaet" ,df_missing_poly_hsp['polymer dH'].isna().sum())
df_missing_poly_hsp_unique_poly = unique_polymer_dataset[['Name','SMILES','dD', 'dP','dH']].copy()

df_without_hsp = df_missing_poly_hsp_unique_poly[df_missing_poly_hsp_unique_poly['dD'].isna()].reset_index(drop=True)
df_without_hsp.to_csv(DATASETS/'raw'/'polymer_without_hsp.csv',index=False)



df_training_dropped_missing_polymer_hsp = w_data.dropna(subset=["polymer dH"]).reset_index(drop=True)
print("Drop missing HSP values for polymers")
df_training_dropped_missing_polymer_hsp.rename(columns={"intensity weighted average over log(Rh (nm))": 'Rh (IW avg log)'}, inplace=True)
print("Done renaming Rh")
# print(df_training_dropped_missing_polymer_hsp['Rh (IW avg log)'])
# df_training_dropped_missing_polymer_hsp.to_pickle(DATASETS/"training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl")
# df_training_dropped_missing_polymer_hsp.to_csv(DATASETS/"training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.csv")

print("Done saving dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl")

Rh_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Rh (IW avg log)'].notna()]
print(len(Rh_data))
Rg_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Rg1 (nm)'].notna()]
Lp_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Lp (nm)'].notna()]
# # print(Rh_data)
# Rh_data.to_pickle(DATASETS/"training_dataset"/"Rh_training_data.pkl")
# Rg_data.to_pickle(DATASETS/"training_dataset"/"Rg_training_data.pkl")
# Lp_data.to_pickle(DATASETS/"training_dataset"/"Lp_training_data.pkl")