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
        # plt.show()
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
        gridsize = max(15, int(max_range / 2))

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
                            bbox={'boxstyle': 'round', 'fc': 'white', 'ec': 'white'})

        # Set axis labels and limits
        g.ax_joint.set_xlim(ax_min, ax_max)
        g.ax_joint.set_ylim(ax_min, ax_max)
        g.set_axis_labels("True Values", "Predicted Values")
        plt.suptitle(f"Parity Plot for {target}", fontweight='bold')
        plt.tight_layout()
        if folder:
            save_img_path(folder, f"Parity Plot {target}.png")
        plt.close()






def process_summary_scores(summary_scores, metric="rmse"):
    data = []
    # Store the set of unique training set sizes for each cluster
    cluster_train_sizes = {}
    
    for cluster, ratios in summary_scores.items():
        cluster_size = ratios.get("Cluster size", 0)  # Get the total size of the cluster
        cluster_train_sizes[cluster] = set()  # Initialize set for each cluster
        
        for ratio, stats in ratios.items():
            if ratio == "Cluster size":
                continue  # Skip the cluster size entry, as it's not a ratio
            
            # Calculate the training set size based on the ratio and cluster size
            train_ratio = float(ratio.replace("ratio_", ""))
            train_set_size = round(float(train_ratio * cluster_size),1)
            # Add the calculated train_set_size to the set of each cluster
            cluster_train_sizes[cluster].add(train_set_size)
            
            metric_test = f"test_{metric}_mean"
            metric_train = f"train_{metric}_mean"
            std_test = f"test_{metric}_std"
            std_train = f"train_{metric}_std"

            if metric_test in stats["test_summary_stats"] and metric_train in stats["train_summary_stats"]:
                data.append({
                    "Cluster": cluster,
                    "Train Set Size": train_set_size,
                    "Score": stats["test_summary_stats"][metric_test],
                    "Std": stats["test_summary_stats"][std_test],
                    "Score Type": "Test"
                })
                data.append({
                    "Cluster": cluster,
                    "Train Set Size": train_set_size,
                    "Score": stats["train_summary_stats"][metric_train],
                    "Std": stats["train_summary_stats"][std_train],
                    "Score Type": "Train"
                })

    return pd.DataFrame(data), cluster_train_sizes

# Function to plot dynamically based on metric and number of clusters
def plot_ood_learning_scores(summary_scores, metric="rmse",folder:Path=None) -> None:
    df, cluster_train_sizes = process_summary_scores(summary_scores, metric)

    if df.empty:
        print(f"No data found for metric '{metric}'.")
        return

    # Find the common training set size across all clusters
    common_train_sizes = set.intersection(*cluster_train_sizes.values())
    if common_train_sizes:
        # If there's a shared training size, take the first one from the set
        shared_train_size = common_train_sizes.pop()
    else:
        # If there's no shared training size, ignore and continue
        shared_train_size = None

    # Determine the number of clusters and set col_wrap dynamically
    num_clusters = df['Cluster'].nunique()
    # col_wrap = min(num_clusters, 4)  # Ensure we don't exceed 4 columns for readability

    g = sns.FacetGrid(df, col="Cluster", col_wrap=num_clusters, height=4, sharey=True)

    def plot_with_shaded_area(data, **kwargs):
        ax = plt.gca()
        sns.lineplot(
            data=data, x="Train Set Size", y="Score", hue="Score Type", 
            style="Score Type", markers=True, ax=ax
        )
        
        for score_type, sub_df in data.groupby("Score Type"):
            # Sort values to ensure correct shading
            sub_df = sub_df.sort_values("Train Set Size")

            ax.fill_between(
                sub_df["Train Set Size"], 
                sub_df["Score"] - sub_df["Std"], 
                sub_df["Score"] + sub_df["Std"], 
                alpha=0.2
            )
        
        # If there's a shared training set size, highlight it on the plot
        # if shared_train_size is not None:
        #     ax.axvline(shared_train_size, color='r', linestyle='--', label="eq train size")
        #     shared_point = data[data["Train Set Size"] == shared_train_size]
        #     if not shared_point.empty:
        #         test_score = shared_point["Score"].values[0]
                
        #         # Find max score for positioning in upper right
        #         max_score = data["Score"].max()
        #         upper_right_x = data["Train Set Size"].max()  # Rightmost x-value
                
        #         ax.annotate(
        #             f"{test_score:.2f}",
        #             xy=(upper_right_x, max_score), 
        #             xycoords="data",
        #             xytext=(20, 20), textcoords="offset points",  # Offset for better visibility
        #             fontsize=12, fontweight="bold",
        #             color="black",
        #             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        #             ha="right", va="top"  # Align text to the top-right
        #         )
        ax.legend(loc="upper left")

    g.map_dataframe(plot_with_shaded_area)
    g.set_axis_labels("Training Set Size", metric.upper())
    g.set_titles(col_template="{col_name}")
    g.add_legend()

    # g._legend.set_bbox_to_anchor((1.01, .48))  # Moves legend outside the plot (right side)
    # g._legend.set_title("Score Type")  # Set legend title


    plt.tight_layout()
    if folder:
            save_img_path(folder, f"learning curve {metric}.png")
    # plt.show()
    plt.close()


def get_residuals_for_learning_curve(data):
    
    pass 
def plot_residual_distribution():


    pass

def plot_residuals_vs_std():
    pass


if __name__ == "__main__":
    # HERE: Path = Path(__file__).resolve().parent
    # results_path = HERE.parent.parent / 'results'/ 'OOD_target_log Rg (nm)'
    # cluster_list = [
    #                 'KM4 ECFP6_Count_512bit cluster',	
    #                 # 'KM3 Mordred cluster',
    #                 # 'substructure cluster',
    #                 # 'EG-Ionic-Based Cluster',
    #                 # 'KM5 polymer_solvent HSP and polysize cluster',
    #                 # 'KM4 polymer_solvent HSP cluster',
    #                 # 'KM4 Mordred_Polysize cluster',
    #                 ]

    # def ensure_long_path(path):
    #     """Ensures Windows handles long paths by adding '\\?\' if needed."""
    #     path_str = str(path)
    #     if os.name == 'nt' and len(path_str) > 250:  # Apply only on Windows if the path is long
    #         return Path(f"\\\\?\\{path_str}")
    #     return path

    # for cluster in cluster_list:
    #     scores_folder_path = results_path / cluster / 'Trimer_scaler'
    #     model = 'NGB'
    #     # score_file = scores_folder_path / f'(Mordred-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_hypOFF_Standard_lc_scores.json'
    #     score_file = scores_folder_path / f'(ECFP3.count.512-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_hypOFF_Standard_lc_scores.json'
    #     score_file = ensure_long_path(score_file)  # Ensure long path support

    #     if not os.path.exists(score_file):
    #         print(f"File not found: {score_file}")
    #         continue  # Skip to the next cluster if the file is missing

    #     with open(score_file, "r") as f:
    #         scores = json.load(f)

    #     saving_folder = scores_folder_path / f'learning curve ({model}_Standard_Mordred_polysize_HSPs_solvent properties)'
    #     plot_ood_learning_scores(scores, metric="rmse", folder=saving_folder)


