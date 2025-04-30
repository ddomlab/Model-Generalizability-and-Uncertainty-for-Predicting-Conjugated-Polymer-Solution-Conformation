import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from matplotlib.ticker import MaxNLocator

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
Vis_path: Path = HERE/'analysis and test' 

def save_path(folder_path:str, file_name:str)->None:
    visualization_folder_path =  folder_path
    os.makedirs(visualization_folder_path, exist_ok=True)    
    fname = file_name
    plt.savefig(visualization_folder_path / fname, dpi=600)
from visualization_setting import set_plot_style

set_plot_style()

training_df_dir: Path = DATASETS/ "training_dataset"

working_data = pd.read_pickle(training_df_dir/'dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl')
working_data = working_data[working_data['Rh (IW avg log)'].notna()]
digitized_df = working_data[working_data['derived Rh (nm)'].notna()]
digitized = digitized_df['Rh (IW avg log)']
table_extraxted_rh_data = working_data[working_data['derived Rh (nm)'].isna()]
table_extraxted_rh = table_extraxted_rh_data['Rh1 (nm)']




flattened_data = pd.DataFrame({
    "Rh (nm)": np.log10(digitized),
    "Source": "Digitized"
})

table_data = pd.DataFrame({
    "Rh (nm)": np.log10(table_extraxted_rh),
    "Source": "Table+Text Reported"
})

combined_data = pd.concat([flattened_data, table_data], ignore_index=True)

combined_data = combined_data.dropna()

plt.figure(figsize=(8, 6))
sns.histplot(data=combined_data, x="Rh (nm)", hue="Source", kde=True,palette="Set2")
plt.xlabel("log Rh (nm)", fontsize=20,fontweight='bold')
plt.ylabel("Count", fontsize=20, fontweight='bold')
plt.title("Distribution of Rh (Digitized vs Table+Text Reported)", fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
save_path(Vis_path,f"Distribution of Rh (Digitized vs Table+Text Reported)")
plt.show()
plt.close()

# print(working_data["derived Rh (nm)"].notna().sum())

# def plot_single_distribution(df,target:str):
#     plt.figure(figsize=(8, 6))
#     sns.histplot(np.log10(df[target]), bins=40, kde=True, color='blue', edgecolor='black')

#     plt.title(f"Distribution of {target}", fontsize=16)
#     plt.xlabel("log (Rh (nm))", fontsize=14)
#     plt.ylabel("Frequency", fontsize=14)
#     plt.xlim(0)
#     plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Adjust `nbins` as needed
#     save_path(Vis_path,f"Distribution of {target}")
#     plt.close()

# plot_single_distribution(working_data, target='Rh (1_1000 nm) (smallest rh)')

# plot_single_distribution()