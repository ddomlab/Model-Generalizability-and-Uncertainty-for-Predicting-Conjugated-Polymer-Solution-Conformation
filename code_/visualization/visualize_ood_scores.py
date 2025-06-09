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


def ensure_long_path(path):
        """Ensures Windows handles long paths by adding '\\?\' if needed."""
        path_str = str(path)
        if os.name == 'nt' and len(path_str) > 250:  
            return Path(f"\\\\?\\{path_str}")
        return path

# def plot_splits_scores(scores: Dict, scores_criteria: List[str], folder:Path=None) -> None:
#     """
#     Plot the scores of the splits with error bars for standard deviation, showing CO_{cluster} and ID_{cluster} scores on the same column.

#     Parameters:
#     scores (dict): Dictionary containing scores for different clusters.
#     scores_criteria (List[str]): List of scoring criteria to plot (e.g., ['mad', 'mae', 'rmse', 'r2', 'std']).
#     """
#     # Extract clusters based on the prefix
#     clusters = [cluster for cluster in scores if cluster.startswith("CO_")]
#     id_clusters = [cluster for cluster in scores if cluster.startswith("ID_")]

#     # Initialize data storage for mean and std values for both CO_ and ID_ clusters
#     data_mean, data_std = {score: [] for score in scores_criteria}, {score: [] for score in scores_criteria}
#     id_data_mean, id_data_std = {score: [] for score in scores_criteria}, {score: [] for score in scores_criteria}

#     # Extract mean and std values for CO_ clusters
#     for score in scores_criteria:
#         for cluster in clusters:
#             mean, std = get_score(scores, cluster, score)
#             data_mean[score].append(mean)
#             data_std[score].append(std)

#     # Extract mean and std values for ID_ clusters
#         for id_cluster in id_clusters:
#             mean, std = get_score(scores, id_cluster, score)
#             id_data_mean[score].append(mean)
#             id_data_std[score].append(std)

#     # Plot each score criterion separately
#     for score in scores_criteria:
#         if all(np.isnan(value) or value == 0 for value in data_mean[score]):
#             continue

#         plt.figure(figsize=(8, 6))

#         # Plot CO_ cluster scores (blue)
#         sns.lineplot(x=clusters, y=data_mean[score], marker="o", linewidth=3, color='blue', label=f"OOD_{score.upper()} Scores",
#                      markersize=10)
#         plt.errorbar(clusters, data_mean[score], yerr=data_std[score], fmt="none", capsize=3, alpha=0.7, color='blue')

#         # Plot ID_ cluster scores (orange)
#         sns.lineplot(x=clusters, y=id_data_mean[score], marker="v", linewidth=3, color='orange', label=f"ID_{score.upper()} Scores",
#                      markersize=10)
#         plt.errorbar(clusters, id_data_mean[score], yerr=id_data_std[score], fmt="none", capsize=3, alpha=0.7, color='orange')

#         # Customize labels, title, and legend
#         plt.ylabel(f"{score.upper()} Score", fontsize=20)
#         plt.xlabel("Clusters", fontsize=20)
#         plt.xticks(rotation=0,fontsize=18)
#         plt.yticks(fontsize=18)
#         plt.title(f"{score.upper()} Score Across Clusters", fontsize=22)
#         plt.legend()
#         plt.tight_layout()
#         if folder:
#             save_img_path(folder, f"Comparitive clusters {score} scores.png")
#         # Display plot
#         # plt.show()
#         plt.close()


# def plot_splits_parity(predicted_values: dict,
#                        ground_truth: dict,
#                        score: dict,
#                        folder: Path) -> None:
#     """
#     Generate parity plots for each target based on predicted values and ground truth.

#     Parameters:
#     - predicted_values (dict): Nested dictionary with structure {target: {cluster: [values]}}
#     - ground_truth (dict): Dictionary with structure {target: [true_values]}
#     - score (dict): Dictionary with structure {target: (r2_avg, r2_stderr)}
#     """

    # seeds = list(next(iter(predicted_values.values())).keys())

    # for target in predicted_values.keys():
    #     if target.startswith("ID_"):
    #         true_values_ext = np.tile(ground_truth.get("ID_y_true", []), len(seeds)) 
    #     else:
    #         true_values_ext = np.tile(ground_truth.get(target, []), len(seeds))  

    #     predicted_values_ext = pd.concat(
    #         [pd.Series(predicted_values[target][col]) for col in seeds],
    #         axis=0, ignore_index=True
    #     )

    #     combined_data = pd.DataFrame({"True Values (nm)": true_values_ext, "Predicted Values (nm)": predicted_values_ext})

    #     range_x = combined_data["True Values (nm)"].max() - combined_data["True Values (nm)"].min()
    #     range_y = combined_data["Predicted Values (nm)"].max() - combined_data["Predicted Values (nm)"].min()
    #     max_range = max(range_x, range_y)
    #     gridsize = max(15, int(max_range / 2))

    #     r2_avg = score[target]["summary_stats"].get(f"test_r2_mean", 0)
    #     r2_stderr = score[target]["summary_stats"].get(f"test_r2_std", 0)

#         r2_avg = score[target]["summary_stats"].get(f"test_rmse_mean", 0)
#         r2_stderr = score[target]["summary_stats"].get(f"test_rmse_std", 0)

        # g = sns.jointplot(
        #     data=combined_data, x="True Values (nm)", y="Predicted Values (nm)",
        #     kind="hex",
        #     joint_kws={"gridsize": gridsize, "cmap": "Blues"},
        #     marginal_kws={"bins": 25}
        # )

        # ax_max = ceil(max(combined_data.max()))
        # ax_min = ceil(min(combined_data.min()))

#         g.ax_joint.plot([0, ax_max], [0, ax_max], ls="--", c=".3")

#         g.ax_joint.annotate(f"$R^2$ = {r2_avg:.2f} ± {r2_stderr:.2f}",
#                             xy=(0.1, 0.9), xycoords='axes fraction',
#                             ha='left', va='center',
#                             bbox={'boxstyle': 'round', 'fc': 'white', 'ec': 'white'})

        # g.ax_joint.set_xlim(ax_min, ax_max)
        # g.ax_joint.set_ylim(ax_min, ax_max)
        # g.set_axis_labels("True Values", "Predicted Values")
        # plt.suptitle(f"Parity Plot for {target}", fontweight='bold')
        # plt.tight_layout()
        # if folder:
        #     save_img_path(folder, f"Parity Plot {target}.png")
        # plt.close()


def plot_ood_parity(prediction: Dict, ground_truth: Dict, 
                    score: Dict = None, folder: Path = None, file_name: str = None) -> None:
    df_results = get_residual_vs_std_full_data(prediction, ground_truth)

    for _, row in df_results.iterrows():
        cluster = row['Cluster']
        predicted_y = row['Predicted y']
        true_y = row['True y']

        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({
            "True Values (nm)": true_y,
            "Predicted Values (nm)": predicted_y
        })

        # Fixed limits and hexbin density
        axis_min, axis_max = 0, 3  # Set to domain-appropriate limits
        gridsize = 20              # Fixed gridsize for consistency
        bins = 20                  # Fixed bin count for marginal histograms

        # Plot with fixed visual parameters
        g = sns.jointplot(
            data=plot_data,
            x="True Values (nm)", y="Predicted Values (nm)",
            kind="hex",
            joint_kws={"gridsize": gridsize, "cmap": "Blues"},
            marginal_kws={"bins": bins}
        )

        g.ax_joint.plot([axis_min, axis_max], [axis_min, axis_max], ls="--", c=".3")
        g.ax_joint.set_xlim(axis_min, axis_max)
        g.ax_joint.set_ylim(axis_min, axis_max)
        g.set_axis_labels("True Values", "Predicted Values")
        plt.suptitle(f"Parity Plot for {cluster}", fontweight='bold')
        plt.tight_layout()

        if folder:
            save_img_path(folder, f"cluster-{cluster}_{file_name}.png")

        # plt.show()
        plt.close()




def get_residual_vs_std_full_data(predicted:Dict,
                                  truth:Dict)->pd.DataFrame:
    
    results = []
    for cluster, preds in predicted.items():
        if cluster.startswith("ID_"):
            continue
        true_values = truth.get(cluster, [])
        residuals = []
        predictions_std = []
        predicted_y = []
        true_y=[]
        for seed, seed_preds in preds.items():
            seed_residuals = np.subtract(seed_preds['y_test_prediction'], true_values)
            residuals.extend(seed_residuals)
            # predictions_std.extend(seed_preds['y_test_uncertainty'])
            predicted_y.extend(seed_preds['y_test_prediction'])
            true_y.extend(true_values)

        results.append([cluster, np.array(residuals), np.array(predicted_y), np.array(true_y)])

    df_results = pd.DataFrame(results, columns=["Cluster", "Residual", "Predicted y", "True y"])
    return df_results

from scipy.stats import pearsonr

# def plot_residual_vs_std_full_data(
#                                 predicted:Dict,
#                                 truth:Dict,
#                                 folder_to_save: Path = None,
#                                 file_name: str = 'NGB_Mordred'
#                                 ) -> None:
#     df = get_residual_vs_std_full_data(predicted, truth)
#     n_clusters = len(df)
#     fig, axes = plt.subplots(1, n_clusters, figsize=(12, 4), sharey=True)

#     if n_clusters == 1:
#         axes = [axes]

#     for ax, (_, row) in zip(axes, df.iterrows()):
#         cluster = row['Cluster']
#         residual = abs(row['Residual'])
#         pred_std = row['Prediction std']
        
#         x_values = residual
#         y_values = pred_std
#         # x_errors = pred_std
#         pearson_r = pearsonr(y_values, x_values)[0]
#         plt.sca(ax)  # Set the current axis so plt.gca() works
#         ax = plt.gca()  # Get current axis

#         ax.scatter(x_values, y_values, color='#1f77b430', edgecolors='none', alpha=0.2, s=80)
#         # ax.errorbar(x_values, y_values, xerr=x_errors, fmt='o', ecolor='blue', elinewidth=2)

#         max_y = max(y_values)
#         max_x = max(x_values)
#         x_line = np.linspace(0, max_x, 100)
#         ax.plot(x_line, x_line, linestyle='--', color='grey', linewidth=1.5, label='y = x')

#         ax.text(
#             0.95, 0.1, 
#             f"Pearson R = {pearson_r:.2f}",
#             transform=ax.transAxes,
#             fontsize=12,
#             horizontalalignment='right',
#             verticalalignment='bottom',
#             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
#         )
#         ax.set_xlabel("Residuals", fontsize=16, fontweight='bold')
#         if ax == axes[0]:
#             ax.set_ylabel("Std of Predictions", fontsize=16, fontweight='bold')
#         ax.set_title(f"{cluster}", fontsize=18)
#         ax.tick_params(axis='both', which='major', labelsize=14)
#         ax.set_ylim(0,max_y+.5)
#     plt.tight_layout()

#     save_img_path(folder_to_save, f"{file_name}.png")
#     plt.show()
#     plt.close()


from typing import Tuple
import pandas as pd

def process_learning_curve_scores(summary_scores,
                                   metric="rmse") -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = []
    cluster_train_sizes = {}

    for cluster, ratios in summary_scores.items():
        if cluster.startswith("CO_") or cluster.startswith("IID_"):
            cluster_size = ratios.get("training size", 0)
            cluster_train_sizes[cluster] = set()
            for ratio, stats in ratios.items():
                if ratio == "training size":
                    continue
                train_ratio = float(ratio.replace("ratio_", ""))
                train_set_size = round(train_ratio * cluster_size, 1)
                cluster_train_sizes[cluster].add(train_set_size)

                metric_test = f"test_{metric}_mean"
                metric_train = f"train_{metric}_mean"
                std_test = f"test_{metric}_std"
                std_train = f"train_{metric}_std"

                if metric_test in stats["test_summary_stats"] and metric_train in stats["train_summary_stats"]:
                    data.append({
                        "Cluster": cluster,
                        "Train Ratio": train_ratio,
                        "Train Set Size": train_set_size,
                        "Score": stats["test_summary_stats"][metric_test],
                        "Std": stats["test_summary_stats"][std_test],
                        "Score Type": "Test"
                    })
                    data.append({
                        "Cluster": cluster,
                        "Train Ratio": train_ratio,
                        "Train Set Size": train_set_size,
                        "Score": stats["train_summary_stats"][metric_train],
                        "Std": stats["train_summary_stats"][std_train],
                        "Score Type": "Train"
                    })

    full_df = pd.DataFrame(data)
    # Determine common train sizes from CO clusters only
    co_only_sizes = {k: v for k, v in cluster_train_sizes.items() if k.startswith("CO_")}
    if co_only_sizes:
        common_train_sizes = set.intersection(*co_only_sizes.values())
    else:
        common_train_sizes = set()

    equal_train_size_df = full_df[full_df["Train Set Size"].isin(common_train_sizes)].copy()

    return full_df, equal_train_size_df





def plot_ood_learning_scores(summary_scores, metric="rmse", folder: Path = None, file_name: str = 'NGB_Mordred') -> None:
    df, _ = process_learning_curve_scores(summary_scores, metric)

    if df.empty:
        print(f"No data found for metric '{metric}'.")
        return

    # common_train_sizes = set.intersection(*cluster_train_sizes.values())
    # shared_train_size = common_train_sizes.pop() if common_train_sizes else None

    num_clusters = df['Cluster'].nunique()
    g = sns.FacetGrid(df, col="Cluster", col_wrap=num_clusters, height=4, sharey=True)

    def plot_with_shaded_area(data, **kwargs):
        ax = plt.gca()
        sns.lineplot(
            data=data, x="Train Set Size", y="Score", hue="Score Type",
            style="Score Type", markers=True, markersize=8, linewidth=2.5, ax=ax
        )

        for score_type, sub_df in data.groupby("Score Type"):
            sub_df = sub_df.sort_values("Train Set Size")
            ax.fill_between(
                sub_df["Train Set Size"],
                sub_df["Score"] - sub_df["Std"],
                sub_df["Score"] + sub_df["Std"],
                alpha=0.2
            )

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        # ax.legend(loc="upper left", fontsize=12, title_fontsize=13)

    g.map_dataframe(plot_with_shaded_area)
    g.set_axis_labels("Training Set Size", metric.upper())

    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(title, fontsize=20)

    plt.tight_layout()
    if folder:
        save_img_path(folder, f"learning curve ({file_name}).png")
    plt.show()
    plt.close()



def plot_bar_ood_iid(data: pd.DataFrame, ml_score_metric: str, 
                    saving_path, file_name: str,
                    figsize=(12, 7), text_size=14,
                    ncol=3, title:Optional[str]=None) -> None:
    
    def parse_cluster_info(cluster_label):
        if cluster_label.startswith("CO_"):
            return cluster_label.split("_")[1], "OOD"
        elif cluster_label.startswith("ID_") or cluster_label.startswith("IID_"):
            return cluster_label.split("_")[1], "IID"
        else:
            return cluster_label, "Unknown"

    parsed_info = data['Cluster'].apply(parse_cluster_info)
    data['BaseCluster'] = parsed_info.apply(lambda x: x[0])
    data['SplitType'] = parsed_info.apply(lambda x: x[1])

    clusters = data['BaseCluster'].unique()
    models = data['Model'].unique()

    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = {model: palette[i] for i, model in enumerate(models)}

    bar_width = 0.4
    n_models = len(models)
    group_width = n_models * 2 * bar_width + 0.5

    fig, ax = plt.subplots(figsize=figsize)

    cluster_ticks = []
    cluster_labels = []

    for i, cluster in enumerate(clusters):
        cluster_offset = i * group_width
        for j, model in enumerate(models):
            subset = data[(data["BaseCluster"] == cluster) & (data["Model"] == model)]
            if subset.empty:
                continue

            base_x = cluster_offset + j * 2 * bar_width
            color = model_colors[model]

            for _, row in subset.iterrows():
                if row["SplitType"] == "OOD":
                    ax.bar(base_x, row["Score"], bar_width, yerr=row["Std"],
                           error_kw={'elinewidth': 1.1, 'capthick': 1.1}, capsize=4,
                           color=color, alpha=0.6, label=f"{model} OOD")
                elif row["SplitType"] == "IID":
                    ax.bar(base_x + bar_width, row["Score"], bar_width, yerr=row["Std"],
                           error_kw={'elinewidth': 1.1, 'capthick': 1.1}, capsize=4,
                           color=color, alpha=1.0, label=f"{model} IID")

        center_of_group = cluster_offset + (n_models * bar_width)
        cluster_ticks.append(center_of_group)
        cluster_labels.append(cluster)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    sub_sufix = '(equal train size)' if 'equal' in saving_path.name else ''
    fig.suptitle(f"OOD vs IID {ml_score_metric.upper()} by Cluster {sub_sufix}", fontsize=text_size + 2, fontweight='bold')

    ax.legend(
        seen.values(), seen.keys(),
        fontsize=text_size - 4,
        frameon=True,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.38),
        ncol=ncol
    )

    # Compute ymax as multiple of 0.2
    max_score = np.nanmax(data["Score"].values)
    ymax = np.ceil(max_score / .2) * .2
    # ax.set_yticks(np.arange(0, ymax+.1, 0.2))
    ax.set_ylim(0, 1.4)

    # Axis labels and ticks
    ax.set_xlabel("Cluster", fontsize=text_size, fontweight='bold')
    ax.set_xticks(cluster_ticks)
    
    ax.set_xticklabels(cluster_labels, fontsize=text_size,rotation=45)
    ax.set_ylabel(ml_score_metric.upper(), fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='y', labelsize=text_size)

    # Layout adjustment

    plt.tight_layout(rect=[0, 0, 1, 1.08])
    save_img_path(saving_path, f"{file_name}.png")
    plt.show()  
    plt.close()




def get_residuals_dist_learning_curve_data(data:Dict) -> pd.DataFrame:
    selected_ratios = {"ratio_0.1", "ratio_0.5", "ratio_0.9"}
    rows = []

    for cluster, ratios in data.items():
        y_ground_truth = np.array(ratios.get("y_true", []))

        for ratio_key, seeds in ratios.items():
            if ratio_key not in selected_ratios:
                continue

            train_ratio = float(ratio_key.replace("ratio_", ""))

            for _, predict_true in seeds.items():
                if 'y_test_pred' in predict_true:
                    residuals = np.array(predict_true['y_test_pred']) - y_ground_truth

                    for res in residuals:
                        rows.append({
                            "Cluster": cluster,
                            "Train Ratio": train_ratio,
                            "Residual": res
                        })
    return pd.DataFrame(rows)


def plot_residual_distribution_learning_curve(predictions: pd.DataFrame, folder_to_save: Path, file_name: str) -> None:
    """
    Plots KDE of residuals for each cluster side-by-side with hue based on train set size.

    Parameters:
    - predictions: DataFrame with columns ['Cluster', 'Train Set Size', 'Residuals']
    - folder_to_save: Path to save the plot
    - file_name: Name of the saved file
    """

    flat_df = get_residuals_dist_learning_curve_data(predictions)
    flat_df["Train Ratio"] = flat_df["Train Ratio"].apply(str)

    clusters = flat_df["Cluster"].unique()
    num_clusters = len(clusters)

    fig, axes = plt.subplots(1, num_clusters, figsize=(11,5), sharey=True)

    if num_clusters == 1:
        axes = [axes]  # make it iterable

    for i, cluster in enumerate(clusters):
        ax = axes[i]
        cluster_df = flat_df[flat_df["Cluster"] == cluster].copy()
        hue_order = sorted(cluster_df["Train Ratio"].unique(), key=lambda x: float(x))

        sns.kdeplot(
            data=cluster_df,
            x="Residual",
            hue="Train Ratio",
            hue_order=hue_order,
            fill=True,
            palette="Set2",
            alpha=0.2,
            linewidth=1,
            ax=ax
        )

        ax.set_title(f"{cluster}", fontsize=22)
        ax.set_xlabel(r"$y_{\text{predict}} - y_{\text{true}}$", fontweight='bold', fontsize=18)
        if i == 0:
            ax.set_ylabel("Distribution Density", fontweight='bold', fontsize=18)
            # sns.move_legend(ax, loc='upper right', title='Train Ratio', title_fontsize=14)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=16)
        ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 0.05, 0.1))
        
        if i != 0:
            ax.get_legend().remove()

    plt.tight_layout()
    save_img_path(folder_to_save, f"{file_name}.png")
    # plt.show()
    plt.close()


from visualize_uncertainty_calibration import get_calibration_confidence_interval, AbsoluteMiscalibrationArea
ama = AbsoluteMiscalibrationArea()

def get_uncertenty_in_learning_curve(pred_file:Dict,method:str)-> pd.DataFrame:
    results = []
    for cluster, ratios in pred_file.items():
        if cluster.startswith("CO_"):
            cluster_size = ratios.get("training size", 0)
            # train_sizes= []
            for ratio, seeds in ratios.items():
                if ratio == "y_true" or ratio == "training size":
                    continue
                train_ratio = float(ratio.replace("ratio_", ""))
                train_set_size = round((train_ratio * cluster_size))
                # train_sizes.append(train_set_size)
                y_predictions= []
                uncertainties = []
                y_true_all = []
                for seed, predictions in seeds.items():
                    y_pred = np.array(predictions.get("y_test_pred", []))
                    y_uncertainty = np.array(predictions.get("y_test_uncertainty", []))
                    y_true = np.array(ratios.get("y_true", []))
                    residual = abs(np.subtract(y_pred, y_true))
                    # y_predictions.extend(y_pred)
                    # uncertainties.extend(y_uncertainty)
                    # y_true_all.extend(y_true)

                    if method=="AMA":
                        uncertainty, _ = get_calibration_confidence_interval(np.array(y_true), np.array(y_pred), np.array(y_uncertainty),
                                                                    ama,n_samples=1)
                    else:
                        uncertainty = pearsonr(np.array(y_uncertainty), np.array(residual))[0]  

                    uncertainties.append(uncertainty)

                    
                results.append({
                    "Cluster": cluster,
                    "Train Set Size": train_set_size,
                    f"{method} mean": np.mean(uncertainties),
                    f"{method} std": np.std(uncertainties),
                })

    return pd.DataFrame(results)


# def plot_ama_vs_train_size(prediction:Dict ,folder:Path=None,file_name:str='NGB_Mordred') -> None:
#     df = get_uncertenty_in_learning_curve(prediction)
#     # print(df)
#     num_clusters = df["Cluster"].nunique()
#     g = sns.FacetGrid(df, col="Cluster", col_wrap=num_clusters, height=4, sharey=True)

#     g.map_dataframe(sns.scatterplot, x="Train Set Size", y="Uncertainty_ama", s=100)  
#     g.map_dataframe(sns.lineplot, x="Train Set Size", y="Uncertainty_ama", linewidth=2.5)

#     g.set_axis_labels("Train Set Size", "Absolute Miscalibration Area")

#     g.set_titles(col_template="{col_name}", fontsize=22)
#     for ax, title in zip(g.axes.flat, g.col_names):
#         ax.set_title(title, fontsize=20)
#     for ax in g.axes.flatten():
#         ax.tick_params(axis='both', which='major', labelsize=16)
#         ax.set_xlabel(ax.get_xlabel(), fontsize=18)
#         ax.set_ylabel(ax.get_ylabel(), fontsize=18)

#     plt.tight_layout()
#     save_img_path(folder, f"Uncertainty vs Train Size ({file_name}).png")
#     # plt.show()
#     plt.close()


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_ood_learning_accuracy_uncertainty(summary_scores: Dict,
                                           prediction: Dict,
                                           metric="rmse",
                                           folder: Path = None,
                                           file_name: str = 'NGB_Mordred',
                                           uncertenty_method: str = "AMA",
                                           title:Optional[str]=None
                                           ) -> None:
    score_df, _ = process_learning_curve_scores(summary_scores, metric)
    unceretainty_df = get_uncertenty_in_learning_curve(prediction, uncertenty_method)

    if score_df.empty:
        print(f"No data found for metric '{metric}'.")
        return

    max_score = np.ceil(score_df["Score"].max() * 10) / 10  
    score_yticks = np.arange(0, max_score + 0.4, 0.4)

    if uncertenty_method == "Pearson R":
        u_yticks = np.arange(-1, 1.5, 0.5)
    else:
        max_uncertainty = np.ceil(unceretainty_df[f'{uncertenty_method} mean'].max() * 10) / 10
        u_yticks = np.arange(0, max_uncertainty + 0.1, 0.1)

    co_clusters = sorted([c for c in score_df['Cluster'].unique() if c.startswith("CO_")])
    g = sns.FacetGrid(score_df[score_df['Cluster'].isin(co_clusters)], col="Cluster", col_wrap=len(co_clusters), height=4, sharey=False)

    twin_axes = []

    def plot_with_dual_y_axis(data, **kwargs):
        cluster = data["Cluster"].iloc[0]
        cluster_idx = cluster.replace("CO_", "")
        id_cluster = f"IID_{cluster_idx}"

        ax = plt.gca()
        ax2 = ax.twinx()
        twin_axes.append((ax, ax2))

        # Plot OOD and IID clusters (Train/Test) in one loop
        for cluster_type, cluster_prefix, line_style, color_map in [
            ("OOD", "CO_", "-", {"Train": "#419BA5", "Test": "#367CE2"}),
            ("ID", "IID_", "--", {"Train": "#E74747", "Test": "#9C529E"})
        ]:
            sub_cluster = f"{cluster_prefix}{cluster_idx}"

            for score_type in ["Train", "Test"]:
                sub_df = score_df[(score_df["Cluster"] == sub_cluster) & (score_df["Score Type"] == score_type)]
                if not sub_df.empty:
                    sub_df = sub_df.sort_values("Train Set Size")
                    color = color_map[score_type]
                    marker = "o" if cluster_type == "OOD" else ("D" if score_type == "Train" else "s")
                    label = f"{cluster_type} {score_type}"

                    sns.lineplot(
                        data=sub_df,
                        x="Train Set Size",
                        y="Score",
                        ax=ax,
                        label=label,
                        linewidth=2.5,
                        marker=marker,
                        linestyle=line_style,
                        color=color
                    )
                    ax.fill_between(
                        sub_df["Train Set Size"],
                        sub_df["Score"] - sub_df["Std"],
                        sub_df["Score"] + sub_df["Std"],
                        alpha=0.2 if cluster_type == "OOD" else 0.15,
                        color=color
                    )

        # Plot uncertainty
        unc_data = unceretainty_df[unceretainty_df["Cluster"] == cluster].sort_values("Train Set Size")
        mean_col = f"{uncertenty_method} mean"
        std_col = f"{uncertenty_method} std"

        ax2.plot(
            unc_data["Train Set Size"],
            unc_data[mean_col],
            color="green",
            marker="*",
            linestyle="--",
            linewidth=2.5,
            markersize=6,
            label=uncertenty_method
        )

        ax2.fill_between(
            unc_data["Train Set Size"],
            unc_data[mean_col] - unc_data[std_col],
            unc_data[mean_col] + unc_data[std_col],
            color="green",
            alpha=0.2
        )

        x_min = 0
        # x_max = int(np.ceil(data["Train Set Size"].max() / 50.0)) * 50
        xticks = np.arange(x_min,  251, 50)
        ax.set_xticks(xticks)
        ax.set_ylim(0, max_score)
        ax.set_yticks(score_yticks)
        ax2.set_ylim(u_yticks.min(), u_yticks.max())
        ax2.set_yticks(u_yticks)
        ax2.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlabel("Training Set Size", fontsize=18, fontweight='bold')

    g.map_dataframe(plot_with_dual_y_axis)

    # Axis labeling cleanup
    for i, (ax, ax2) in enumerate(twin_axes):
        if i == 0:
            ax.set_ylabel(metric.upper(), fontsize=18, color="#4487D0", fontweight='bold')
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if i == len(twin_axes) - 1:
            ax2.set_ylabel(uncertenty_method, fontsize=18, color="green", fontweight='bold')
        else:
            ax2.set_ylabel("")
            ax2.set_yticklabels([])

        ax.set_title(g.col_names[i], fontsize=20)

    # Shared legend
    custom_lines = [
        Line2D([0], [0], color='#419BA5', lw=3.5, marker='o', label='OOD Train'),
        Line2D([0], [0], color='#367CE2', lw=3.5, marker='o', label='OOD Test'),
        Line2D([0], [0], color='#E74747', lw=3.5, marker='D', linestyle='--', label='ID Train'),
        Line2D([0], [0], color='#9C529E', lw=3.5, marker='s', linestyle='--', label='ID Test'),
        Line2D([0], [0], color='green', lw=3.5, marker='*', linestyle='--', label='OOD Test')
    ]

    if title:
        g.fig.suptitle(title, fontsize=24, fontweight='bold', y=1.12)

    plt.figlegend(
        handles=custom_lines,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        fontsize=18,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if folder:
        save_img_path(folder, f"learning curve ({file_name}).png")
    # plt.show()
    plt.close()




# comparison_of_features_lc = {
#     '(Mordred-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_hypOFF_Standard_lc':'Xn + + Mordred ',
#     '(Mordred-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_RF_hypOFF_Standard_lc':'Mordred+continuous+aging',
#     '(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_hypOFF_Standard_lc':'continuous',
#     '(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_RF_hypOFF_Standard_lc':'continuous+aging',
# }

def get_comparison_of_features(model: str, suffix: str):
    return {
        f'(concentration-temperature-solvent dP-solvent dD-solvent dH)_{model}{suffix}': 'solvent_properties + solvent_HSPs',
        f'(concentration-temperature-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_{model}{suffix}': 'solvent_properties + solvent_HSPs + environmental.thermal history',
        f'(light exposure-aging time-aging temperature-prep temperature-prep time)_{model}{suffix}': 'environmental.thermal history',
        f'(Xn)_{model}{suffix}': 'Xn',
        f'(Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize',
        f'(Xn-Mw-PDI-polymer dP-polymer dD-polymer dH)_{model}{suffix}': 'Xn + polysize + polymer HSPs',
        f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_{model}{suffix}': 'combination of all',
        f'(Mordred-Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize + Mordred',
        f'(MACCS-Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize + MACCS',
        f'(ECFP3.count.512-Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize + ECFP6.count.512',
    }

# Example usage:


if __name__ == "__main__":
    HERE: Path = Path(__file__).resolve().parent
    results_path = HERE.parent.parent / 'results'/ 'OOD_target_log Rg (nm)'
    cluster_list = [
                    # 'KM4 ECFP6_Count_512bit cluster',	
                    # 'KM3 Mordred cluster',
                    # 'HBD3 MACCS cluster',
                    # 'KM5 polymer_solvent HSP and polysize cluster',
                    # 'KM4 polymer_solvent HSP and polysize cluster',
                    'substructure cluster',
                    'KM4 polymer_solvent HSP cluster',
                    # 'KM4 Mordred_Polysize cluster',
                    # 'Polymers cluster'
                    ]



    for cluster in cluster_list:
        # scores_folder_path = results_path / cluster / 'Trimer_scaler'
        # for fp in ['MACCS', 'Mordred']:
        #     all_score_eq_training_size = []
        #     for model in ['RF']:

                # suffix = '_v1_(max_feat_sqrt)'
                # suffix = ''
                # score_file_lc = scores_folder_path / f'({fp}-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_hypOFF_Standard_lc{suffix}_scores.json'
                # predictions_file_lc = scores_folder_path / f'({fp}-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_hypOFF_Standard_lc{suffix}_predictions.json'
                # truth_file_full = scores_folder_path / f'({fp}-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_Standard_ClusterTruth.json'
                # predictions_full = scores_folder_path / f'({fp}-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_Standard_predictions.json'
                # score_file_lc = ensure_long_path(score_file_lc)  # Ensure long path support
                # predictions_file_lc = ensure_long_path(predictions_file_lc)
                # predictions_full = ensure_long_path(predictions_full)
                # truth_file_full = ensure_long_path(truth_file_full)
                # if not os.path.exists(score_file_lc):
                #     print(f"File not found: {score_file_lc}")
                #     continue  



                #     # NGB XGB learning curve
                # with open(score_file_lc, "r") as f:
                #     scores_lc = json.load(f)

                # with open(predictions_file_lc, "r") as s:
                #     predictions_lc = json.load(s)

                # saving_folder_lc_score = scores_folder_path / f'learning curve'
                # plot_ood_learning_scores(scores_lc, metric="rmse", folder=saving_folder_lc_score, file_name=f'{model}_{fp}')
                # print("Save learning curve scores")
                # saving_uncertainty = scores_folder_path / f'uncertainty'
                # if model == 'XGBR':
                #     continue
        #         # print(predictions_file_lc)
        #         plot_ama_vs_train_size(predictions_lc, saving_uncertainty, file_name=f'{model}_{fp}')
        #         print("Save learning curve uncertainty")

                # uncertenty + score in learning curve
                # saving_uncertainty = scores_folder_path / f'uncertainty_score'
                # plot_ood_learning_accuracy_uncertainty(scores_lc, predictions_lc, metric="rmse",
                #                                         folder=saving_uncertainty, file_name=f'{model}_{fp}',uncertenty_method="Pearson R")
                # print("Save learning curve scores and uncertainty")

        # Plot uncertenty + score in learning curve for comparison of features
        model = 'RF'
        comparison_of_features_lc = get_comparison_of_features(model, "_hypOFF_Standard_lc")

        all_score_eq_training_size = []
        for file, file_discription in comparison_of_features_lc.items():

            if any(keyword in file for keyword in ['Mordred', 'MACCS', 'ECFP3.count.512']):
                scores_folder_path = results_path / cluster / 'Trimer_scaler'
            else:
                scores_folder_path = results_path / cluster / 'scaler'
            
            score_file_lc = ensure_long_path(scores_folder_path / f'{file}_scores.json')
            predictions_file_lc = ensure_long_path(scores_folder_path / f'{file}_predictions.json')

            if not os.path.exists(score_file_lc) or not os.path.exists(predictions_file_lc):
                print(f"File not found: {file}")
                continue  

            with open(score_file_lc, "r") as f:
                scores_lc = json.load(f)
            with open(predictions_file_lc, "r") as s:
                predictions_lc = json.load(s)

            # saving_folder_lc_score_of_features = results_path / cluster / f'comparison of features learning curve'            
            # plot_ood_learning_accuracy_uncertainty(scores_lc, predictions_lc, metric="rmse",
            #                                         folder=saving_folder_lc_score_of_features,
            #                                         file_name=f'{file_discription}',uncertenty_method="Pearson R",
            #                                         title=file_discription)
            # print("Save learning curve scores and uncertainty")

            # Plot OOD vs IID barplot at the same training size for comparison of features
            
            _, ood_iid_eq_tarining_size_df = process_learning_curve_scores(scores_lc, metric="rmse")
            ood_iid_eq_tarining_size_df = ood_iid_eq_tarining_size_df[ood_iid_eq_tarining_size_df["Score Type"] == "Test"]
            ood_iid_eq_tarining_size_df['Model'] = file_discription
            all_score_eq_training_size.append(ood_iid_eq_tarining_size_df)
        all_score_eq_training_size: pd.DataFrame = pd.concat(all_score_eq_training_size, ignore_index=True)
        saving_folder = results_path / cluster / f'OOD-IID bar plot at equal training set (comparison of features)'
        plot_bar_ood_iid(all_score_eq_training_size, 'rmse', 
                         saving_folder, file_name=f'rmse_{model}_comparison of feature',
                             text_size=16, figsize=(15, 8), ncol=3)

        print("save OOD vs IID bar plot at equal training size for comparison of features")

                # plot OOD vs IID barplot at the same training size 

            #     _, ood_iid_eq_tarining_size_df = process_learning_curve_scores(scores_lc, metric="rmse")
            #     ood_iid_eq_tarining_size_df = ood_iid_eq_tarining_size_df[ood_iid_eq_tarining_size_df["Score Type"] == "Test"]
            #     ood_iid_eq_tarining_size_df['Model'] = model

            #     all_score_eq_training_size.append(ood_iid_eq_tarining_size_df)
            # all_score_eq_training_size:pd.DataFrame = pd.concat(all_score_eq_training_size, ignore_index=True)
            # saving_folder = scores_folder_path/  f'OOD-IID bar plot at equal training set (RF_max_feat_sqrt)'
            # plot_bar_ood_iid(all_score_eq_training_size, 'rmse', saving_folder, file_name=f'numerical-{fp}_metric-rmse_noraml',
            #                  text_size=16,figsize=(8, 6))
            # print("save OOD vs IID bar plot at equal training size")

                # residual distribution
                # saving_folder = scores_folder_path / f'KDE of residuals'
                # plot_residual_distribution_learning_curve(predictions_lc, saving_folder, file_name=f'{model}_{fp}')


                # Plot residual vs std (uncertenty):
                # with open(predictions_full, "r") as f:
                #     predictions_data = json.load(f)

                # with open(truth_file_full, "r") as f:
                #     truth_data = json.load(f)
                # saving_folder = scores_folder_path / f'residual vs std (uncertainty)'
                # plot_residual_vs_std_full_data(predictions_data, truth_data, saving_folder, file_name=f'{model}_{fp}')

        # RF learning curve
        # scores_folder_path = results_path / cluster / 'scaler'
        # score_file_lc_RF = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_hypOFF_Standard_lc_scores.json'
        # prediction_file_lc_RF = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_hypOFF_Standard_lc_predictions.json'
        # prediction_file_full = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_Standard_predictions.json'
        # truth_file_full = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_Standard_ClusterTruth.json'
        # prediction_file_full = ensure_long_path(prediction_file_full)  
        # truth_file_full = ensure_long_path(truth_file_full)
        # score_file_lc_RF = ensure_long_path(score_file_lc_RF)
        # prediction_file_lc_RF = ensure_long_path(prediction_file_lc_RF)
        # if not os.path.exists(score_file_lc_RF):
        #     print(f"File not found: {score_file_lc_RF}")
        #     continue 

        # with open(score_file_lc_RF, "r") as f:
        #     scores_lc = json.load(f)
        # with open(prediction_file_lc_RF, "r") as f:
        #     predictions_lc = json.load(f)

        # ama vs train size in lc
        # saving_folder_lc_score = scores_folder_path / f'learning curve'
        # plot_ood_learning_scores(scores_lc, metric="rmse", folder=saving_folder_lc_score, file_name='RF_scaler')
        # saving_uncertainty = scores_folder_path / f'uncertainty'
        # plot_ama_vs_train_size(predictions_lc,saving_uncertainty, file_name=f'RF_scaler')


        # saving_uncertainty = scores_folder_path / f'uncertainty_score'
        # plot_ood_learning_accuracy_uncertainty(scores_lc, predictions_lc, metric="rmse", folder=saving_uncertainty, 
        #                                      file_name=f'RF_scaler_PearsonR',uncertenty_method="Pearson R")
        # print("Save learning curve scores and uncertainty")


        # # residual distribution in lc
        # saving_folder = scores_folder_path / f'KDE of residuals'
        # plot_residual_distribution_learning_curve(predictions_lc, saving_folder, file_name=f'RF_scaler')

        

        # Plot residual vs std (uncertenty):

        # with open(prediction_file_full, "r") as f:
        #     predictions_data = json.load(f)

        # with open(truth_file_full, "r") as f:
        #     truth_data = json.load(f)

        # saving_folder = scores_folder_path / f'residual vs std (uncertainty)'
        # plot_residual_vs_std_full_data(predictions_data, truth_data, saving_folder, file_name=f'RF_scaler')
                # if not os.path.exists(predictions_full) or not os.path.exists(truth_file_full):
                #     print(f"File not found: {predictions_full} or {truth_file_full}")
                #     continue  
                # with open(predictions_full, "r") as s:
                #     predictions_data = json.load(s)

                # with open(truth_file_full, "r") as f:
                #     truth_data = json.load(f)
                # # #Plot parity plot
                # saving_folder = scores_folder_path / f'Parity plot'/ f'{fp}'
                # plot_ood_parity(predictions_data, truth_data, folder=saving_folder, file_name=f'{model}_{fp}')







