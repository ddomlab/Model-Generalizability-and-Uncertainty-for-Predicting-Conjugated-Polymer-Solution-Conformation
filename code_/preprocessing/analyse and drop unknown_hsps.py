import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.pkl"
unique_polymer_dir :Path = DATASETS/'raw'/'SMILES_to_BigSMILES_Conversion_wo_block_copolymer_with_HSPs.xlsx'
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

# Collect data
data_summary = {
    'Target': [],
    'Original': [],
    'After Filter': [],
    'Reduced': []
}

for target in targets:
    original = df_missing_poly_hsp[target].notna().sum()
    after_filter = df_missing_poly_hsp[target].notna() & df_missing_poly_hsp['polymer dH'].notna()
    after = after_filter.sum()
    reduced = original - after

    data_summary['Target'].append(target.split()[0])  # Clean label: 'Rh', 'Rg', 'Lp'
    data_summary['Original'].append(original)
    data_summary['After Filter'].append(after)
    data_summary['Reduced'].append(reduced)

# Convert to DataFrame
df_plot = pd.DataFrame(data_summary)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.25
x = range(len(df_plot))

ax.bar([i - width for i in x], df_plot['Original'], width=width, label='Original', color='skyblue')
ax.bar(x, df_plot['After Filter'], width=width, label='After Filter', color='lightgreen')
ax.bar([i + width for i in x], df_plot['Reduced'], width=width, label='Reduced', color='salmon')

ax.set_xticks(x)
ax.set_xticklabels(df_plot['Target'], fontsize=14)
ax.set_ylabel('Number of Data Points', fontsize=14)
ax.set_title('Data Reduction by Missing Polymer dH Values', fontsize=16)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()







if __name__ == "__main__":
       df_missing_poly_hsp_unique_poly = unique_polymer_dataset[['Name','SMILES','dD', 'dP','dH']].copy()

       df_without_hsp = df_missing_poly_hsp_unique_poly[df_missing_poly_hsp_unique_poly['dD'].isna()].reset_index(drop=True)
       df_without_hsp.to_csv(DATASETS/'raw'/'polymer_without_hsp.csv',index=False)
       print("Unique polymers without HSP values saved to polymer_without_hsp.csv!")



       df_training_dropped_missing_polymer_hsp = w_data.dropna(subset=["polymer dH"]).reset_index(drop=True)
       print("Drop missing HSP values for polymers")
       print(df_training_dropped_missing_polymer_hsp['Rh (IW avg log)'])
       df_training_dropped_missing_polymer_hsp.to_pickle(DATASETS/"training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl")
       print("Done saving dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl")

       Rh_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Rh (IW avg log)'].notna()]
       Rg_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Rg1 (nm)'].notna()]
       Lp_data = df_training_dropped_missing_polymer_hsp[df_training_dropped_missing_polymer_hsp['Lp (nm)'].notna()]
