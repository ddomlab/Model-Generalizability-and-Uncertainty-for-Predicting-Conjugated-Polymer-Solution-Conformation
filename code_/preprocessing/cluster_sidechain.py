import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem
# from split_Rh_peaks import save_path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from visualization.visualization_setting import (set_plot_style,
                                                 save_img_path)
set_plot_style(tick_size=18)

HERE: Path = Path(__file__).resolve().parent
VISUALIZATION = HERE.parent/ "visualization"
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal (40-1000 nm)_added.pkl"
w_data = pd.read_pickle(training_df_dir)




alkyl_smarts_patterns = []
for i in range(2, 15):  # From C2 (ethyl) to C15 (dodecyl)
    alkyl_smarts_patterns.append('[CX4]' * i)  # Alkyl chain with i carbons

ether_smarts = Chem.MolFromSmarts('[OX2;!r]')
ester_smarts = Chem.MolFromSmarts('[C;!r](=O)[O;!r][C;!r]')
ionic_smarts = Chem.MolFromSmarts('[+,-]')  # Matches charged atoms (ionic)
alkyl_cyanide = Chem.MolFromSmarts('[N;!r]CC') 

def determine_side_chain_type(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if not mol:
            return "Invalid SMILES"

        if mol.HasSubstructMatch(ionic_smarts):
            return "ionic"

        elif mol.HasSubstructMatch(ether_smarts) or mol.HasSubstructMatch(ester_smarts):
            return "polar"

        if mol.HasSubstructMatch(alkyl_cyanide):
            return 'polar'

        for alkyl_smarts in alkyl_smarts_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(alkyl_smarts)):
                return "non-polar"
        

        return "polar"
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return "Error"
    




def plot_peak_distribution(data:pd.DataFrame, target:str,column_to_draw):
    df= data.dropna(subset=[target]).reset_index()
    if type(df[target].loc[1]) == list or type(df[target].loc[1]) == np.array:
        reordered_df = pd.DataFrame(df[target].tolist(), columns=["First Peak", "Second Peak", "Third Peak"])
        df = pd.concat([df, reordered_df], axis=1)

    # melted_df = reordered_df.melt(var_name="Peak Position", value_name="Value")
    # melted_df['transformed_value'] = np.log10(melted_df['Value'])
    fig, ax = plt.subplots(figsize=(10, 8))
    if (df[column_to_draw] <= 0).any():
        # print("Warning: Non-positive values found in the column. Filtering them out.")
        df = df[df[column_to_draw] > 0]
    sns.histplot(data=df, x=np.log10(df[column_to_draw]), kde=True, hue="Side Chain type", bins=40, ax=ax)
    x_label = f'log (Rh {column_to_draw})' if 'Peak' in column_to_draw else f'log ({column_to_draw})'

    ax.set_xlabel(x_label)
    ax.set_ylabel('Occurrence')
    ax.set_title(f'Distribution of  {column_to_draw} over side chain type')
    # ax.tick_params(axis='x', labelsize=25)  # Set font size for x-axis ticks
    # ax.tick_params(axis='y', labelsize=25)  # Set font size for y-axis ticks
    
    box_inset = ax.inset_axes([0.01, -0.4, 0.99, 0.2])  
    sns.boxplot(x=np.log10(df[column_to_draw]), data=df, hue="Side Chain type", ax=box_inset)
    box_inset.set(yticks=[], xlabel=None)
    box_inset.legend_.remove()
    # box_inset.tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    # plt.show()
    fname = f"Distribution of {column_to_draw} over side chain type"
    fname = fname.replace("/", "_").replace("\\", "_").replace(":", "_")
    print(fname)
    save_img_path(VISUALIZATION/"analysis and test",f"{fname}.png")
    plt.close()



if __name__ == "__main__":
    Rh_data = w_data[w_data["multimodal Rh"].notna()]
    Rh_data["Side Chain type"] = Rh_data["Monomer SMILES"].apply(determine_side_chain_type)

    Rg_data = w_data[w_data['Rg1 (nm)'].notna()]
    Rg_data["Side Chain type"] = Rg_data["Monomer SMILES"].apply(determine_side_chain_type)

    unique_data = w_data[['canonical_name', 'Monomer SMILES']].drop_duplicates()
    unique_data["Side Chain type"] = unique_data["Monomer SMILES"].apply(determine_side_chain_type)
    w_data["Side Chain type"] = w_data["Monomer SMILES"].apply(determine_side_chain_type)
    
    for group in ['non-polar', 'polar', 'ionic']:
        # print(f"{group}:",len(w_data[w_data['Side Chain type']==group]))
        # print(f"unique {group}:",len(unique_data[unique_data['Side Chain type']==group]))
        print(f"Rg: {group}:",len(Rg_data[Rg_data['Side Chain type']==group]))
        print(f"Rh: {group}:",len(Rh_data[Rh_data['Side Chain type']==group]))


    # plot_peak_distribution(Rg_data,'Rg1 (nm)','Rg1 (nm)')
    # plot_peak_distribution(Rh_data,'multimodal Rh',"First Peak")
    # plot
    # plot_peak_distribution(Rh_data,'multimodal Rh',"Second Peak")    
    # plot_peak_distribution(Rh_data,'multimodal Rh',"Third Peak")
    # plot_peak_distribution(Rg_data,'Rg1 (nm)', 'Temperature SANS/SLS/DLS/SEC (K)')
    # plot_peak_distribution(Rg_data,'Rg1 (nm)','Rg1 (nm)')
    # plot_peak_distribution(Rg_data,'Rg1 (nm)', 'Ra')
    plot_peak_distribution(Rg_data,'Rg1 (nm)', 'Concentration (mg/ml)')


