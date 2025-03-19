import json
from itertools import product
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import os 
# import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import ceil
from visualization_setting import set_plot_style, save_img_path
from scipy.stats import spearmanr



set_plot_style()

HERE: Path = Path(__file__).resolve().parent
VISUALIZATION = HERE.parent/ "visualization"
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "Rg data with clusters.pkl"
w_data = pd.read_pickle(training_df_dir)



import numpy as np
import matplotlib.pyplot as plt

DP = w_data['DP'].values
log_DP = np.log10(DP)  # Compute log10(DP) directly
Rg = w_data['Rg1 (nm)'].values
log_Rg = w_data['log Rg (nm)'].values  # Experimental log Rg

# Generate smooth DP values for the theoretical curve
DP_fine = np.logspace(np.log10(min(DP)), np.log10(max(DP)), 100)  # Corrected logspace
log_DP_fine = np.log10(DP_fine)  # Log-transformed DP values
log_Rg_theory = 0.5 * log_DP_fine  # Theoretical log Rg = 0.5 * log DP

spearman_corr, _ = spearmanr(Rg, DP)

plt.figure(figsize=(7, 5))
plt.scatter(log_DP, log_Rg, color='#4682B4', label='Data')  # Scatter plot of experimental data
plt.plot(log_DP_fine, log_Rg_theory, color='grey', linestyle='--', label=r'$\log R_g = 0.5 \log DP$')  # Theoretical line

# Labels and formatting
plt.xlabel(r'$\log_{10} DP$', fontsize=18)
plt.ylabel(r'$\log_{10} R_g$ (nm)', fontsize=18)
plt.legend(fontsize=12)
plt.xlim(1,4)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
