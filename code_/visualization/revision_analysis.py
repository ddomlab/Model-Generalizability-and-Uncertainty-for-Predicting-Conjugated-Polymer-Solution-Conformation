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

import sys

sys.modules.setdefault("numpy._core",         np.core)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.umath",   np.core.umath)

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
JSONS =  DATASETS/'json_resources'
VISUALIZATION = HERE.parent/ "visualization"
training_df_dir: Path = DATASETS/ "training_dataset"/"non_imputed_full_Rg_data.pkl"
training_df_dir_imputed: Path = DATASETS/ "training_dataset"/"Rg data with clusters aging imputed.pkl"
w_data = pd.read_pickle(training_df_dir)
w_data_imputed = pd.read_pickle(training_df_dir_imputed)

unique_pairs_count = w_data[['Xn']].drop_duplicates().shape[0]
print("Number of unique (Xn) pairs:", unique_pairs_count)


unique_pairs_count = w_data[["Concentration (mg/ml)", "Temperature SANS/SLS/DLS/SEC (K)",
                             'solvent dP', 'solvent dD', 'solvent dH',
                              "Dark/light", "Aging time (hour)", "To Aging Temperature (K)",
                              "Sonication/Stirring/heating Temperature (K)", "Merged Stirring /sonication/heating time(min)"]].drop_duplicates().shape[0]
print("Number of unique environmental condition pairs:", unique_pairs_count)


print(len(w_data_imputed["Trimer_Mordred"].loc[0]))
print(len(w_data_imputed["Trimer_MACCS"].loc[0]))
# print(w_data_imputed["Trimer_ECFP"])
# Group by Xn and compute mean, std, RSD for log Rg
# group_stats = (
#     w_data_imputed.groupby('Xn')['log Rg (nm)']
#     .agg(['mean', 'std', 'min', 'max'])
#     .rename(columns={'mean': 'mean_val', 'std': 'std_val'})
# )

# # Compute RSD (%)
# group_stats['rsd (%)'] = (group_stats['std_val'] / group_stats['mean_val']) * 100
# group_stats["range"] = group_stats["max"] - group_stats["min"]


# plt.figure(figsize=(5,3.5))
# sns.histplot(group_stats['rsd (%)'], bins=30, color="#107b93")
# plt.xlabel("RSD (%)", fontsize=14)
# plt.ylabel("Occurrence", fontsize=14)
# # plt.title("Distribution of RSD of log Rg (nm) across Xn groups", fontsize=16)
# plt.tight_layout()
# plt.ylim(0, 15)
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

# below_30 = w_data[w_data["Concentration (mg/ml)"] < 30]
# print(below_30)
# sns.histplot(np.log(below_30["Concentration (mg/ml)"]), bins=30, color="#7b1093")
# plt.show(

# features = ['Xn', 'Mw (g/mol)', 'PDI', 'Concentration (mg/ml)', 'Temperature SANS/SLS/DLS/SEC (K)',
#                              "polymer dP", "polymer dD", "polymer dH", 'solvent dP', 'solvent dD', 'solvent dH',
#                               "Dark/light", "Aging time (hour)", "To Aging Temperature (K)",
#                               "Sonication/Stirring/heating Temperature (K)", "Merged Stirring /sonication/heating time(min)"]

# print(w_data["Aging time (hour)"].isna().sum())
# print(w_data[features].info())




# ==========================
# 1. Your Data
# ==========================

# Means
# mean_data = {
#     "mean":      [0.20, 0.19],
#     "iterative": [0.25, 0.23],
#     "uniform KNN_3": [0.19, 0.17],
#     "uniform KNN_4": [0.20, 0.17],
#     "uniform KNN_5": [0.20, 0.17],
#     "uniform KNN_6": [0.20, 0.18],
#     "uniform KNN_7": [0.20, 0.18],
#     "distance KNN_3": [0.21, 0.18],
#     "distance KNN_4": [0.21, 0.18],
#     "distance KNN_5": [0.21, 0.19],
#     "distance KNN_6": [0.21, 0.20],
#     "distance KNN_7": [0.21, 0.20],
# }

# # STDs
# std_data = {
#     "mean":      [0.05, 0.05],
#     "iterative": [0.04, 0.04],
#     "uniform KNN_3": [0.04, 0.07],
#     "uniform KNN_4": [0.06, 0.07],
#     "uniform KNN_5": [0.04, 0.06],
#     "uniform KNN_6": [0.04, 0.05],
#     "uniform KNN_7": [0.05, 0.06],
#     "distance KNN_3": [0.05, 0.07],
#     "distance KNN_4": [0.05, 0.06],
#     "distance KNN_5": [0.05, 0.06],
#     "distance KNN_6": [0.05, 0.07],
#     "distance KNN_7": [0.05, 0.07],

# }

# models = ["RF", "XGB"]

# df_mean = pd.DataFrame(mean_data, index=models)
# df_std  = pd.DataFrame(std_data, index=models)

# # ==========================
# # 2. Annotation text: mean on first line, ±std on next line
# # ==========================

# annot = (
#     df_mean.round(2).astype(str)
#     + "\n± "
#     + df_std.round(2).astype(str)
# )

# # ==========================
# # 3. Plot Heatmap
# # ==========================

# plt.figure(figsize=(11, 4.5))
# heatmap = sns.heatmap(
#     df_mean,
#     annot=annot,
#     fmt="",
#     cmap="viridis",
#     annot_kws={"size": 14},
#     cbar_kws={"label": "Average RMSE ± Stdev"},
#     vmax=.6, vmin=.1
# )

# # Rotate colorbar label vertically
# heatmap.collections[0].colorbar.ax.set_ylabel(
#     "Average RMSE ± Stdev",
#     rotation=270,     # text runs top → bottom
#     labelpad=15,
#     fontsize=14
# )
# plt.xticks(rotation=45)

# # plt.title("Model Score Heatmap (mean ± std)", fontsize=14)
# plt.xlabel("imputation Method")
# plt.ylabel("Model")

# plt.tight_layout()
# save_img_path(VISUALIZATION / "analysis and test", "Model_performance_iterative_vs_mean_revision.png")
# plt.show()
