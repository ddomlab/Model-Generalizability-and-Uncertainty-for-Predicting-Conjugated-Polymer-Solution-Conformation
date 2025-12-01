import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import json
from visualization_setting import save_img_path, set_plot_style, save_img_path
from matplotlib.ticker import MaxNLocator
import os
import seaborn as sns
set_plot_style()


HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
JSONS =  DATASETS/'json_resources'
VISUALIZATION = HERE.parent/ "visualization"
training_df_dir: Path = DATASETS/ "training_dataset"/"non_imputed_full_Rg_data.pkl"
w_data = pd.read_pickle(training_df_dir)


unique_pairs_count = w_data[['Xn']].drop_duplicates().shape[0]
print("Number of unique (Xn) pairs:", unique_pairs_count)


unique_pairs_count = w_data[["Concentration (mg/ml)", "Temperature SANS/SLS/DLS/SEC (K)",
                             'solvent dP', 'solvent dD', 'solvent dH',
                              "Dark/light", "Aging time (hour)", "To Aging Temperature (K)",
                              "Sonication/Stirring/heating Temperature (K)", "Merged Stirring /sonication/heating time(min)"]].drop_duplicates().shape[0]
print("Number of unique environmental condition pairs:", unique_pairs_count)




# Group by Xn and compute mean, std, RSD for log Rg
# group_stats = (
#     w_data.groupby('Xn')['log Rg (nm)']
#     .agg(['mean', 'std', 'min', 'max'])
#     .rename(columns={'mean': 'mean_val', 'std': 'std_val'})
# )

# # Compute RSD (%)
# group_stats['rsd (%)'] = (group_stats['std_val'] / group_stats['mean_val']) * 100
# group_stats["range"] = group_stats["max"] - group_stats["min"]


# plt.figure(figsize=(5,3.5))
# sns.histplot(group_stats['rsd (%)'], bins=30, color="#107b93")
# plt.xlabel("RSD (%)", fontsize=14)
# plt.ylabel("Count", fontsize=14)
# # plt.title("Distribution of RSD of log Rg (nm) across Xn groups", fontsize=16)
# plt.tight_layout()
# save_img_path(VISUALIZATION / "analysis and test", "RSD_distribution_log_Rg_across_Xn_groups.png")
# plt.show()


# plt.figure(figsize=(5,3.5))
# sns.histplot(group_stats['range'], bins=30, color="#10937b")
# plt.xlabel("Range log(nm)", fontsize=14)
# plt.ylabel("Count", fontsize=14)
# # plt.title("Distribution of RSD of log Rg (nm) across Xn groups", fontsize=16)
# plt.tight_layout()
# save_img_path(VISUALIZATION / "analysis and test", "range_distribution_log_Rg_across_Xn_groups.png")
# plt.show()

below_30 = w_data[w_data["Concentration (mg/ml)"] < 30]
print(below_30)
sns.histplot(np.log(below_30["Concentration (mg/ml)"]), bins=30, color="#7b1093")
plt.show()
