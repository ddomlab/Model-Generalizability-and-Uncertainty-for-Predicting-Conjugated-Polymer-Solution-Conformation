import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem
from matplotlib.ticker import MaxNLocator

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
    print(df[column_to_draw].notna().sum())
    print(df[column_to_draw])
    sns.histplot(data=df, x=np.log10(df[column_to_draw]), kde=True, hue="Side Chain type", bins=20, ax=ax)
    x_label = f'log (Rh {column_to_draw})' if 'Peak' in column_to_draw else f'log ({column_to_draw})'

    ax.set_xlabel(x_label)
    ax.set_ylabel('Occurrence')
    ax.set_title(f'Distribution of  {column_to_draw} over side chain type in {target}')
    # ax.tick_params(axis='x', labelsize=25)  # Set font size for x-axis ticks
    # ax.tick_params(axis='y', labelsize=25)  # Set font size for y-axis ticks
    
    box_inset = ax.inset_axes([0.01, -0.4, 0.99, 0.2])  
    sns.boxplot(x=np.log10(df[column_to_draw]), data=df, hue="Side Chain type", ax=box_inset)
    box_inset.set(yticks=[], xlabel=None)
    box_inset.legend_.remove()
    # box_inset.tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    # plt.show()
    fname = f"Distribution of {column_to_draw} over side chain type in {target}"
    fname = fname.replace("/", "_").replace("\\", "_").replace(":", "_")
    save_img_path(VISUALIZATION/"analysis and test",f"{fname}.png")
    plt.close()

def plot_hanson_space(df, hsp_material:str):

    dP = df[f"{hsp_material} dP"]
    dD = df[f"{hsp_material} dD"]
    dH = df[f"{hsp_material} dH"]

    dP_freq = dP.map(dP.value_counts())
    # dD_freq = dD.map(dD.value_counts())
    # dH_freq = dH.map(dH.value_counts())

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    maxlocator = {
        "integer":True,
        "prune":"lower",
        "nbins":6,
                  }
    
    sc = axes[0].scatter(dP, dD, c=dP_freq, cmap="viridis")
    axes[0].set_xlabel(r'$\delta P$ (MPa$^{1/2}$)')
    axes[0].set_ylabel(r'$\delta D$ (MPa$^{1/2}$)')
    axes[0].xaxis.set_major_locator(MaxNLocator(**maxlocator))
    axes[0].yaxis.set_major_locator(MaxNLocator(**maxlocator))


    # Scatter plot for dP vs dH
    sc = axes[1].scatter(dP, dH, c=dP_freq, cmap="viridis")
    axes[1].set_xlabel(r'$\delta P$ (MPa$^{1/2}$)')
    axes[1].set_ylabel(r'$\delta H$ (MPa$^{1/2}$)')
    axes[1].xaxis.set_major_locator(MaxNLocator(**maxlocator))
    axes[1].yaxis.set_major_locator(MaxNLocator(**maxlocator))


    # Scatter plot for dD vs dH
    sc = axes[2].scatter(dD, dH, c=dP_freq, cmap="viridis")
    axes[2].set_xlabel(r'$\delta D$ (MPa$^{1/2}$)')
    axes[2].set_ylabel(r'$\delta H$ (MPa$^{1/2}$)')
    axes[2].xaxis.set_major_locator(MaxNLocator(**maxlocator))
    axes[2].yaxis.set_major_locator(MaxNLocator(**maxlocator))
    cbar3 = fig.colorbar(sc, ax=axes[2], orientation="vertical", shrink=0.7, aspect=25)
    cbar3.set_label("Frequency")

    fig.suptitle(f"{hsp_material} Hansen Solubility Parameters", fontsize=14, fontweight="bold")

    plt.tight_layout()
    # plt.show()
    fname = f"Scattering of {hsp_material} Hansen Solubility Parameters"
    save_img_path(VISUALIZATION/"analysis and test",f"{fname}.png")
    plt.close()








if __name__ == "__main__":
    Rh_data = w_data[w_data["multimodal Rh"].notna()]
    # print(len(Rh_data['Temperature SANS/SLS/DLS/SEC (K)'].notna()))
    # Rh_data["Side Chain type"] = Rh_data["Monomer SMILES"].apply(determine_side_chain_type)

    Rg_data = w_data[w_data['Rg1 (nm)'].notna()]
    # Rg_data["Side Chain type"] = Rg_data["Monomer SMILES"].apply(determine_side_chain_type)

    # unique_data = w_data[['canonical_name', 'Monomer SMILES']].drop_duplicates()
    # unique_data["Side Chain type"] = unique_data["Monomer SMILES"].apply(determine_side_chain_type)
    # w_data["Side Chain type"] = w_data["Monomer SMILES"].apply(determine_side_chain_type)
    
    # for group in ['non-polar', 'polar', 'ionic']:
    #     # print(f"{group}:",len(w_data[w_data['Side Chain type']==group]))
    #     # print(f"unique {group}:",len(unique_data[unique_data['Side Chain type']==group]))
    #     print(f"Rg: {group}:",len(Rg_data[Rg_data['Side Chain type']==group]))
    #     print(f"Rh: {group}:",len(Rh_data[Rh_data['Side Chain type']==group]))


    # plot_peak_distribution(Rg_data,'Rg1 (nm)','Rg1 (nm)')
    # plot_peak_distribution(Rh_data,'multimodal Rh',"First Peak")
    # plot
    # plot_peak_distribution(Rh_data,'multimodal Rh',"Second Peak")    
    # plot_peak_distribution(Rh_data,'multimodal Rh',"Third Peak")
    # plot_peak_distribution(Rg_data,'Rg1 (nm)', 'Temperature SANS/SLS/DLS/SEC (K)')
    # plot_peak_distribution(Rg_data,'Rg1 (nm)','Rg1 (nm)')
    # plot_peak_distribution(Rg_data,'Rg1 (nm)', 'Ra')
    # features = ['Temperature SANS/SLS/DLS/SEC (K)','Ra', 'Concentration (mg/ml)', 'Mn (g/mol)']
    # for feats in features:
    #     plot_peak_distribution(Rh_data,'multimodal Rh', feats)
    
    plot_hanson_space(Rg_data,'solvent')