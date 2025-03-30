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



set_plot_style()


def get_score(scores: Dict, cluster: str, score_metric: str) -> float:
    """
    Helper function to get the mean or std score for a given cluster and score type.
    Returns 0 if not available.
    """
    mean_key = f"test_{score_metric}_mean"
    std_key = f"test_{score_metric}_std"
    return (
        scores.get(cluster, {}).get("summary_stats", {}).get(mean_key, 0),
        scores.get(cluster, {}).get("summary_stats", {}).get(std_key, 0),
    )



def plot_splits_scores(scores: Dict, scores_criteria: List[str], folder:Path=None) -> None:
    """
    Plot the scores of the splits with error bars for standard deviation, showing CO_{cluster} and ID_{cluster} scores on the same column.

    Parameters:
    scores (dict): Dictionary containing scores for different clusters.
    scores_criteria (List[str]): List of scoring criteria to plot (e.g., ['mad', 'mae', 'rmse', 'r2', 'std']).
    """
    # Extract clusters based on the prefix
    clusters = [cluster for cluster in scores if cluster.startswith("CO_")]
    id_clusters = [cluster for cluster in scores if cluster.startswith("ID_")]

    # Initialize data storage for mean and std values for both CO_ and ID_ clusters
    data_mean, data_std = {score: [] for score in scores_criteria}, {score: [] for score in scores_criteria}
    id_data_mean, id_data_std = {score: [] for score in scores_criteria}, {score: [] for score in scores_criteria}

    # Extract mean and std values for CO_ clusters
    for score in scores_criteria:
        for cluster in clusters:
            mean, std = get_score(scores, cluster, score)
            data_mean[score].append(mean)
            data_std[score].append(std)

    # Extract mean and std values for ID_ clusters
        for id_cluster in id_clusters:
            mean, std = get_score(scores, id_cluster, score)
            id_data_mean[score].append(mean)
            id_data_std[score].append(std)

    # Plot each score criterion separately
    for score in scores_criteria:
        if all(np.isnan(value) or value == 0 for value in data_mean[score]):
            continue

        plt.figure(figsize=(8, 6))

        # Plot CO_ cluster scores (blue)
        sns.lineplot(x=clusters, y=data_mean[score], marker="o", linewidth=3, color='blue', label=f"OOD_{score.upper()} Scores",
                     markersize=10)
        plt.errorbar(clusters, data_mean[score], yerr=data_std[score], fmt="none", capsize=3, alpha=0.7, color='blue')

        # Plot ID_ cluster scores (orange)
        sns.lineplot(x=clusters, y=id_data_mean[score], marker="v", linewidth=3, color='orange', label=f"ID_{score.upper()} Scores",
                     markersize=10)
        plt.errorbar(clusters, id_data_mean[score], yerr=id_data_std[score], fmt="none", capsize=3, alpha=0.7, color='orange')

        # Customize labels, title, and legend
        plt.ylabel(f"{score.upper()} Score", fontsize=20)
        plt.xlabel("Clusters", fontsize=20)
        plt.xticks(rotation=0,fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(f"{score.upper()} Score Across Clusters", fontsize=22)
        plt.legend()
        plt.tight_layout()
        if folder:
            save_img_path(folder, f"Comparitive clusters {score} scores.png")
        # Display plot
        plt.show()
        plt.close()


def plot_splits_parity(predicted_values: dict,
                       ground_truth: dict,
                       score: dict,
                       folder: Path) -> None:
    """
    Generate parity plots for each target based on predicted values and ground truth.

    Parameters:
    - predicted_values (dict): Nested dictionary with structure {target: {cluster: [values]}}
    - ground_truth (dict): Dictionary with structure {target: [true_values]}
    - score (dict): Dictionary with structure {target: (r2_avg, r2_stderr)}
    """
    # Extract clusters (seeds), assuming all targets share the same clusters
    seeds = list(next(iter(predicted_values.values())).keys())

    for target in predicted_values.keys():
        # Check if the target starts with 'ID_' and use 'ID_y_true' for all such targets
        if target.startswith("ID_"):
            true_values_ext = np.tile(ground_truth.get("ID_y_true", []), len(seeds))  # Use 'ID_y_true' for all 'ID_' targets
        else:
            true_values_ext = np.tile(ground_truth.get(target, []), len(seeds))  # Use the target directly for non-'ID_' targets

        # Flatten predicted values for each target
        predicted_values_ext = pd.concat(
            [pd.Series(predicted_values[target][col]) for col in seeds],
            axis=0, ignore_index=True
        )

        # Combine true and predicted values into a DataFrame
        combined_data = pd.DataFrame({"True Values (nm)": true_values_ext, "Predicted Values (nm)": predicted_values_ext})

        range_x = combined_data["True Values (nm)"].max() - combined_data["True Values (nm)"].min()
        range_y = combined_data["Predicted Values (nm)"].max() - combined_data["Predicted Values (nm)"].min()
        max_range = max(range_x, range_y)
        gridsize = max(5, int(max_range / 2))

        # Get R² statistics
        r2_avg = score[target]["summary_stats"].get(f"test_r2_mean", 0)
        r2_stderr = score[target]["summary_stats"].get(f"test_r2_std", 0)

        # Create the parity plot
        g = sns.jointplot(
            data=combined_data, x="True Values (nm)", y="Predicted Values (nm)",
            kind="hex",
            joint_kws={"gridsize": gridsize, "cmap": "Blues"},
            marginal_kws={"bins": 25}
        )

        # Determine plot limits
        ax_max = ceil(max(combined_data.max()))
        ax_min = ceil(min(combined_data.min()))

        # Add y=x reference line
        g.ax_joint.plot([0, ax_max], [0, ax_max], ls="--", c=".3")

        # Annotate with R² value
        g.ax_joint.annotate(f"$R^2$ = {r2_avg:.2f} ± {r2_stderr:.2f}",
                            xy=(0.1, 0.9), xycoords='axes fraction',
                            ha='left', va='center',
                            bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

        # Set axis labels and limits
        g.ax_joint.set_xlim(ax_min, ax_max)
        g.ax_joint.set_ylim(ax_min, ax_max)
        g.set_axis_labels("True Values", "Predicted Values")
        plt.suptitle(f"Parity Plot for {target}", fontweight='bold')
        plt.tight_layout()
        if folder:
            save_img_path(folder, f"Parity Plot {target}.png")
        plt.close()