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

from visualization_setting import set_plot_style, save_img_path



set_plot_style()

def plot_splits_scores(scores: Dict, scores_criteria: List[str], folder: Path)->None:
    """
    Plot the scores of the splits with error bars for standard deviation.

    Parameters:
    scores (dict): Dictionary containing scores for different clusters.
    scores_criteria (List[str]): List of scoring criteria to plot (e.g., ['mad', 'mae', 'rmse', 'r2', 'std']).
    """
    # Extract relevant clusters
    clusters = [cluster for cluster in scores.keys() if cluster.startswith("CO_")]

    # Initialize data storage
    data_mean = {score: [] for score in scores_criteria}
    data_std = {score: [] for score in scores_criteria}

    # Extract mean and std values
    for cluster in clusters:
        for score in scores_criteria:
            mean_key = f"test_{score}_mean"
            std_key = f"test_{score}_std"
            data_mean[score].append(scores[cluster]["summary_stats"].get(mean_key, 0))
            data_std[score].append(scores[cluster]["summary_stats"].get(std_key, 0))

    # Plot each score criterion separately
    for score in scores_criteria:
        if all(np.isnan(value) or value == 0 for value in data_mean[score]):
            continue
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=clusters, y=data_mean[score], marker="o", linewidth=1.5)
        plt.errorbar(clusters, data_mean[score], yerr=data_std[score], fmt="none", capsize=3, alpha=0.7)

        # Labels and title
        plt.ylabel(f"{score.upper()} Score")
        plt.xlabel("Clusters")
        plt.xticks(rotation=0)
        plt.title(f"{score.upper()} Score Across Clusters")
        if folder:
            save_img_path(folder/'comparitive cluster scores', f"Comparitive clusters {score} scores.png")
        plt.close()



def plot_splits_parity():
    """
    Plot the parity of the splits
    """
    pass