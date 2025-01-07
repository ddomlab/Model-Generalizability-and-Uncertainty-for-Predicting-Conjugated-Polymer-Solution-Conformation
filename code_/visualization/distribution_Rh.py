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
training_df_dir: Path = DATASETS/ "training_dataset"

working_data = pd.read_pickle(training_df_dir/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended.pkl")

flattened_df = working_data.explode("Rh at peaks (above 1 nm)", ignore_index=True)

flattened_df["Rh at peaks (above 1 nm)"] = pd.to_numeric(flattened_df["Rh at peaks (above 1 nm)"])

plt.figure(figsize=(8, 6))

# Create the histogram
sns.histplot(np.log10(flattened_df["Rh at peaks (above 1 nm)"]), bins=40, kde=True, color='blue', edgecolor='black')

# Set title and axis labels
plt.title("Distribution of Rh at peaks (above 1 nm)", fontsize=16)
plt.xlabel('log (Rh (nm))', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Increase the number of ticks on the x-axis
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=25))  # Adjust `nbins` as needed

plt.show()