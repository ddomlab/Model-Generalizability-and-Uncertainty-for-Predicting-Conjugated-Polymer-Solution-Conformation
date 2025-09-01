import json
from itertools import product
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import os 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from math import ceil
from visualization_setting import set_plot_style, save_img_path
from visualize_uncertainty_calibration import (compute_residual_error_cal, 
                                               compute_ama,gaussian_nll, 
                                               compute_cv,
                                               compute_all_uncertainty_metrics)

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

    palette = sns.color_palette("Set2", n_colors=len(models))
    model_colors = {model: palette[i] for i, model in enumerate(models)}

    bar_width = 0.45
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
                           error_kw={'elinewidth': 1, 'capthick': 1}, capsize=3,
                           color=color, alpha=1, label=f"{model} OOD")
                elif row["SplitType"] == "IID":
                    ax.bar(base_x + bar_width, row["Score"], bar_width, yerr=row["Std"],
                           error_kw={'elinewidth': 1, 'capthick': 1}, capsize=3,
                           color=color, alpha=.5, label=f"{model} IID")

        center_of_group = cluster_offset + (n_models * bar_width)
        cluster_ticks.append(center_of_group)
        cluster_labels.append(cluster)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    # sub_sufix = '(equal train size)' if 'equal' in saving_path.name else ''
    # fig.suptitle(f"OOD vs IID {ml_score_metric.upper()} by Cluster {sub_sufix}", fontsize=text_size + 2, fontweight='bold')

    ax.legend(
        seen.values(), seen.keys(),
        fontsize=text_size - 2,
        frameon=True,
        loc='upper left',
        # bbox_to_anchor=(0.5, 1.12),
        ncol=ncol
    )

    # Compute ymax as multiple of 0.2
    max_score = np.nanmax(data["Score"].values)
    ymax = np.ceil(max_score / .2) * .2
    # ax.set_yticks(np.arange(0, ymax+.1, 0.2))
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.4 , .2))
    # Axis labels and ticks
    ax.set_xlabel("Cluster", fontsize=text_size, fontweight='bold')
    ax.set_xticks(cluster_ticks)

    ax.set_xticklabels(cluster_labels, fontsize=text_size, rotation=45)
    ax.set_ylabel(f'{ml_score_metric.upper()}', fontsize=text_size, fontweight='bold')
    ax.tick_params(axis='y', labelsize=text_size)

    # Layout adjustment

    plt.tight_layout()
    save_img_path(saving_path, f"{file_name}.png")
    plt.show()  
    plt.close()


# def plot_bar_ood_iid(data: pd.DataFrame, ml_score_metric: str, 
#                     saving_path, file_name: str,
#                     figsize=(12, 7), text_size=14,
#                     ncol=3, title:Optional[str]=None) -> None:
    
#     def parse_cluster_info(cluster_label):
#         if cluster_label.startswith("CO_"):
#             return cluster_label.split("_")[1], "OOD"
#         elif cluster_label.startswith("ID_") or cluster_label.startswith("IID_"):
#             return cluster_label.split("_")[1], "IID"
#         else:
#             return cluster_label, "Unknown"

#     parsed_info = data['Cluster'].apply(parse_cluster_info)
#     data['BaseCluster'] = parsed_info.apply(lambda x: x[0])
#     data['SplitType'] = parsed_info.apply(lambda x: x[1])

#     clusters = data['BaseCluster'].unique()
#     models = data['Model'].unique()

#     palette = sns.color_palette("Set2", n_colors=len(models))
#     model_colors = {model: palette[i] for i, model in enumerate(models)}

#     bar_width = 0.4
#     n_models = len(models)
#     group_width = n_models * 2 * bar_width + 0.5

#     # Separate polar cluster from others
#     polar_cluster = "Polar"
#     nonpolar_clusters = [c for c in clusters if c != polar_cluster]

#     # ---- Create two side-by-side axes (shared y-axis) ----
#     fig, (ax_main, ax_polar) = plt.subplots(
#         1, 2, figsize=figsize, gridspec_kw={'width_ratios':[4,1]}, sharey=True
#     )

#     # ---- Plot non-polar clusters ----
#     cluster_ticks = []
#     cluster_labels = []
#     for i, cluster in enumerate(nonpolar_clusters):
#         cluster_offset = i * group_width
#         for j, model in enumerate(models):
#             subset = data[(data["BaseCluster"] == cluster) & (data["Model"] == model)]
#             if subset.empty:
#                 continue
#             base_x = cluster_offset + j * 2 * bar_width
#             color = model_colors[model]

#             for _, row in subset.iterrows():
#                 xpos = base_x if row["SplitType"]=="OOD" else base_x+bar_width
#                 alpha = 1 if row["SplitType"]=="OOD" else 0.5
#                 ax_main.bar(xpos, row["Score"], bar_width, yerr=row["Std"],
#                            error_kw={'elinewidth': 1.1, 'capthick': 1.1}, capsize=4,
#                            color=color, alpha=alpha, label=f"{model} {row['SplitType']}")

#         center_of_group = cluster_offset + (n_models * bar_width)
#         cluster_ticks.append(center_of_group)
#         cluster_labels.append(cluster)

#     ax_main.set_xticks(cluster_ticks)
#     ax_main.set_xticklabels(cluster_labels, fontsize=text_size)
#     fig.supxlabel("Cluster", fontsize=text_size, fontweight="bold")

#     # ---- Plot polar cluster on right axis ----
#     for j, model in enumerate(models):
#         subset = data[(data["BaseCluster"] == polar_cluster) & (data["Model"] == model)]
#         if subset.empty:
#             continue
#         base_x = j * 2 * bar_width
#         color = model_colors[model]

#         for _, row in subset.iterrows():
#             xpos = base_x if row["SplitType"]=="OOD" else base_x+bar_width
#             alpha = 1 if row["SplitType"]=="OOD" else 0.5
#             ax_polar.bar(xpos, row["Score"], bar_width, yerr=row["Std"],
#                          error_kw={'elinewidth': 1.1, 'capthick': 1.1}, capsize=4,
#                          color=color, alpha=alpha)

#     ax_polar.set_xticks([bar_width])
#     ax_polar.set_xticklabels([polar_cluster], fontsize=text_size)
#     # ax_polar.set_xlabel("Cluster", fontsize=text_size, fontweight='bold')

#     # ---- Shared Y-axis ----
#     ax_main.set_ylabel(f'{ml_score_metric.upper()}', fontsize=text_size, fontweight='bold')
#     ax_main.set_ylim(0, 1)
#     ax_main.set_yticks(np.arange(0, 1.4, .2))
#     ax_main.tick_params(axis='y', labelsize=text_size)
#     ax_polar.tick_params(axis='y', labelsize=text_size)

#     # ---- Legend (deduplicated) ----
#     handles, labels = ax_main.get_legend_handles_labels()
#     seen = {}
#     for h, l in zip(handles, labels):
#         seen[l] = h
#     ax_main.legend(seen.values(), seen.keys(), fontsize=text_size-4, frameon=True,
#                    loc='upper center', ncol=ncol)

#     plt.tight_layout()
#     save_img_path(saving_path, f"{file_name}.png")
#     plt.show()
#     plt.close()


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


def get_uncertenty_in_learning_curve(pred_file: Dict, method: Optional[str] = None) -> pd.DataFrame:
    results = []

    for cluster, ratios in pred_file.items():
        if not cluster.startswith("CO_"):
            continue

        cluster_size = ratios.get("training size", 0)
        y_true = np.array(ratios.get("y_true", []))

        for ratio_key, seeds in ratios.items():
            if ratio_key in {"y_true", "training size"}:
                continue

            train_ratio = float(ratio_key.replace("ratio_", ""))
            train_set_size = round(train_ratio * cluster_size)

            all_metrics = []

            for seed, predictions in seeds.items():
                y_pred = np.array(predictions.get("y_test_pred", []))
                y_uncertainty = np.array(predictions.get("y_test_uncertainty", []))

                metrics = compute_all_uncertainty_metrics(y_true, y_pred, y_uncertainty, method=method)
                all_metrics.append(metrics)

            # Aggregate
            if method:
                metric_values = np.array(all_metrics)
                results.append({
                    "Cluster": cluster,
                    "Train Set Size": train_set_size,
                    f"{method} mean": np.mean(metric_values),
                    f"{method} std": np.std(metric_values),
                })
            else:
                # Compute mean/std for each metric in the returned dictionary
                aggregated = {"Cluster": cluster, "Train Set Size": train_set_size}
                keys = all_metrics[0].keys()
                for k in keys:
                    vals = [m[k] for m in all_metrics]
                    aggregated[f"{k} mean"] = np.mean(vals)
                    aggregated[f"{k} std"] = np.std(vals)
                results.append(aggregated)

    return pd.DataFrame(results)


def plot_ood_learning_uncertainty(
    model_predictions: Dict[str, Dict],
    folder: Path = None,
    file_name: str = 'model_comparison',
    title: Optional[str] = None,
    fontsize: int = 18,
    uncertainty_methods: Tuple[str, str] = ("Spearman R", "Cv")
) -> None:
    method1, method2 = uncertainty_methods
    color_1 = '#E74747'  # For method1
    color_2 = '#367CE2'  # For method2

    model_styles = {
        'RF': '-',     # solid
        'XGBR': '--',  # dashed
        'NGB': '-.'    # dash-dot (optional, you can use '--' if only 2 models)
    }

    model_markers = {'RF': 'o', 'XGBR': '^', 'NGB': 'D'}

    all_df_1, all_df_2 = [], []

    for model_name, prediction in model_predictions.items():
        df_1 = get_uncertenty_in_learning_curve(prediction, method=method1)
        df_1["Model"] = model_name
        all_df_1.append(df_1)

        df_2 = get_uncertenty_in_learning_curve(prediction, method=method2)
        df_2["Model"] = model_name
        all_df_2.append(df_2)

    df_1 = pd.concat(all_df_1, ignore_index=True)
    df_2 = pd.concat(all_df_2, ignore_index=True)

    co_clusters = sorted([c for c in df_1['Cluster'].unique() if c.startswith("CO_")])
    col_wrap = 4 if len(co_clusters) > 4 else len(co_clusters)

    g = sns.FacetGrid(df_1[df_1['Cluster'].isin(co_clusters)],
                      col="Cluster", col_wrap=col_wrap, height=4, sharey=False)
    twin_axes = []

    def plot_model_uncertainties(data, **kwargs):
        cluster = data["Cluster"].iloc[0]
        ax = plt.gca()
        ax2 = ax.twinx()
        twin_axes.append((ax, ax2))

        for model_name in data["Model"].unique():
            marker = model_markers.get(model_name, 'o')
            line_style = model_styles.get(model_name, '-')

            data_1 = df_1[(df_1["Cluster"] == cluster) & (df_1["Model"] == model_name)].sort_values("Train Set Size")
            data_2 = df_2[(df_2["Cluster"] == cluster) & (df_2["Model"] == model_name)].sort_values("Train Set Size")

            ax.plot(
                data_1["Train Set Size"],
                data_1[f"{method1} mean"],
                label=None,
                marker=marker,
                linewidth=2.5,
                linestyle=line_style,
                color=color_1,
                markersize=8
            )
            ax.fill_between(
                data_1["Train Set Size"],
                data_1[f"{method1} mean"] - data_1[f"{method1} std"],
                data_1[f"{method1} mean"] + data_1[f"{method1} std"],
                alpha=0.2,
                color=color_1,
            )

            ax2.plot(
                data_2["Train Set Size"],
                data_2[f"{method2} mean"],
                label=None,
                marker=marker,
                linewidth=2.5,
                linestyle=line_style,
                color=color_2,
                markersize=8
            )
            ax2.fill_between(
                data_2["Train Set Size"],
                data_2[f"{method2} mean"] - data_2[f"{method2} std"],
                data_2[f"{method2} mean"] + data_2[f"{method2} std"],
                alpha=0.2,
                color=color_2
            )

        xticks = np.arange(0, 251, 50)
        ax.set_xticks(xticks)
        ax.set_ylim(-1, 1)
        ax.set_yticks(np.arange(-1, 1.5, 0.5))
        ax2.set_ylim(0, .5)
        ax2.set_yticks(np.arange(0, .6, 0.1))

        ax.set_xlabel("Training Set Size", fontsize=fontsize, fontweight='bold')
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        ax2.tick_params(axis='y', labelsize=fontsize - 2)

    g.map_dataframe(plot_model_uncertainties)

    for i, (ax, ax2) in enumerate(twin_axes):
        if i == 0:
            ax.set_ylabel(method1, fontsize=fontsize, color=color_1, fontweight='bold')
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if i == len(twin_axes) - 1:
            ax2.set_ylabel(method2, fontsize=fontsize, color=color_2, fontweight='bold')
        else:
            ax2.set_ylabel("")
            ax2.set_yticklabels([])

        ax.set_title(g.col_names[i], fontsize=fontsize + 2)

    # Legend for models (using linestyle instead of color)
    legend_lines = [
        Line2D([0], [0], color='black', marker=model_markers[model], linestyle=model_styles[model], markersize=8,label=model)
        for model in df_1["Model"].unique()
    ]

    if title:
        g.fig.suptitle(title, fontsize=fontsize + 6, fontweight='bold', y=1.12)

    plt.figlegend(
        handles=legend_lines,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(legend_lines),
        fontsize=fontsize,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if folder:
        save_img_path(folder, f"uncertainty comparison ({file_name}).png")

    plt.show()
    plt.close()







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
        raise ValueError(f"No data found for metric '{metric}' in the learning curve scores.")

    max_score = np.ceil(score_df["Score"].max() * 10) / 10  
    score_yticks = np.arange(0, max_score + 0.4, 0.4)

    if uncertenty_method == "Spearman R":
        u_yticks = np.arange(-1, 1.5, 0.5)

    elif uncertenty_method == "AMA":
        u_yticks = np.arange(0, .6, 0.1)
        
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
    plt.show()
    plt.close()

# all models
def plot_ood_learning_accuracy_only(
    all_scores: Dict[str, Dict],
    metric: str = "rmse",
    folder: Path = None,
    file_name: str = 'model_accuracy_comparison',
    title: Optional[str] = None,
    fontsize: int = 14
) -> None:
    set2_palette = sns.color_palette("Set2")

    model_colors = {
        'RF': set2_palette[1],
        'XGBR': set2_palette[0],
        'NGB': set2_palette[2],
    }

    # Style mapping
    domain_styles = {
        "CO_": "-",      # OOD: solid
        "IID_": "--",    # IID: dashed
    }

    score_type_markers = {
        "Train": 'o',
        "Test": '^'
    }

    all_dfs = []
    for model_name, summary_scores in all_scores.items():
        score_df, _ = process_learning_curve_scores(summary_scores, metric)
        score_df["Model"] = model_name
        all_dfs.append(score_df)

    score_df = pd.concat(all_dfs, ignore_index=True)

    if score_df.empty:
        raise ValueError(f"No data found for metric '{metric}'.")

    max_score = np.ceil(score_df["Score"].max() * 10) / 10
    score_yticks = np.arange(0, max_score + 0.4, 0.4)

    co_clusters = sorted(set(c.replace("CO_", "") for c in score_df['Cluster'].unique() if c.startswith("CO_")))
    col_wrap = 4 if len(co_clusters) > 4 else len(co_clusters)

    g = sns.FacetGrid(score_df[score_df['Cluster'].str.startswith("CO_")],
                      col="Cluster", col_wrap=col_wrap, height=4, sharey=False)

    def plot_accuracy(data, **kwargs):
        cluster_oid = data["Cluster"].iloc[0]
        cluster_idx = cluster_oid.replace("CO_", "")
        ax = plt.gca()

        for model in data["Model"].unique():
            for prefix in ["CO_", "IID_"]:
                for score_type in ["Train", "Test"]:
                    cluster_name = f"{prefix}{cluster_idx}"
                    sub_df = score_df[
                        (score_df["Cluster"] == cluster_name) &
                        (score_df["Model"] == model) &
                        (score_df["Score Type"] == score_type)
                    ]

                    if sub_df.empty:
                        continue

                    sub_df = sub_df.sort_values("Train Set Size")
                    color = model_colors.get(model, 'black')
                    linestyle = domain_styles[prefix]
                    marker = score_type_markers[score_type]

                    ax.plot(
                        sub_df["Train Set Size"],
                        sub_df["Score"],
                        label=None,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=2.5,
                        color=color,
                        markersize=8
                    )
                    ax.fill_between(
                        sub_df["Train Set Size"],
                        sub_df["Score"] - sub_df["Std"],
                        sub_df["Score"] + sub_df["Std"],
                        alpha=0.15,
                        color=color
                    )

        ax.set_xticks(np.arange(0, 251, 50))
        ax.set_ylim(0, max_score)
        ax.set_yticks(score_yticks)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_xlabel("Training Set Size", fontsize=fontsize + 2, fontweight='bold')

        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)

    g.map_dataframe(plot_accuracy)

    for i, ax in enumerate(g.axes.flat):
        if i == 0:
            ax.set_ylabel(metric.upper(), fontsize=fontsize + 2, fontweight='bold')
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.set_title(g.col_names[i], fontsize=fontsize + 4)

    # Legend
    color_legend = [
        Line2D([0], [0], color=color, lw=3, label=model)
        for model, color in model_colors.items()
        if model in score_df["Model"].unique()
    ]
    domain_legend = [
        Line2D([0], [0], color='black', lw=2.5, linestyle=linestyle, label=label)
        for label, linestyle in zip(["OOD", "IID"], ["-", "--"])
    ]
    score_type_legend = [
        Line2D([0], [0], color='black', marker=marker, linestyle='None', markersize=10, label=label)
        for label, marker in score_type_markers.items()
    ]

    if title:
        g.fig.suptitle(title, fontsize=fontsize + 10, fontweight='bold', y=1.12)

    plt.figlegend(
        handles=color_legend + domain_legend + score_type_legend,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(color_legend + domain_legend + score_type_legend),
        fontsize=fontsize,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if folder:
        save_img_path(folder, f"accuracy comparison ({file_name}).png")

    plt.show()
    plt.close()


# def plot_ood_learning_accuracy_only(
#     all_scores: Dict[str, Dict],
#     metric: str = "rmse",
#     folder: Path = None,
#     file_name: str = 'model_accuracy_comparison',
#     title: Optional[str] = None,
#     mode_to_show: Optional[List[str]] = 'RF',
#     fontsize: int = 14
# ) -> None:
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
#     import pandas as pd
#     from matplotlib.lines import Line2D

#     # Color map for OOD vs IID
#     color_map = {
#         'CO_': "#B93C4D",   # OOD
#         'IID_': "#4A6FB4",  # IID
#     }

#     # Markers only depend on Train/Test now
#     marker_map = {
#         'Train': 'o',
#         'Test': '^'
#     }

#     all_dfs = []

#     for model_name, summary_scores in all_scores.items():
#         if model_name != mode_to_show:
#             continue
#         score_df, _ = process_learning_curve_scores(summary_scores, metric)
#         score_df["Model"] = model_name
#         all_dfs.append(score_df)

#     score_df = pd.concat(all_dfs, ignore_index=True)

#     if score_df.empty:
#         raise ValueError(f"No data found for metric '{metric}'.")

#     max_score = np.ceil(score_df["Score"].max() * 10) / 10
#     score_yticks = np.arange(0, max_score + 0.4, 0.4)

#     co_clusters = sorted(set(c.replace("CO_", "") for c in score_df['Cluster'].unique() if c.startswith("CO_")))
#     col_wrap = 4 if len(co_clusters) > 4 else len(co_clusters)

#     g = sns.FacetGrid(score_df[score_df['Cluster'].str.startswith("CO_")],
#                       col="Cluster", col_wrap=col_wrap, height=4, sharey=False)

#     def plot_accuracy(data, **kwargs):
#         cluster_oid = data["Cluster"].iloc[0]
#         cluster_idx = cluster_oid.replace("CO_", "")
#         ax = plt.gca()

#         for prefix in ["CO_", "IID_"]:
#             for score_type in ["Train", "Test"]:
#                 cluster_name = f"{prefix}{cluster_idx}"
#                 subset = score_df[
#                     (score_df["Cluster"] == cluster_name) &
#                     (score_df["Model"] == "RF") &
#                     (score_df["Score Type"] == score_type)
#                 ]

#                 if subset.empty:
#                     continue

#                 subset = subset.sort_values("Train Set Size")
#                 color = color_map[prefix]
#                 marker = marker_map[score_type]
#                 linestyle = '--' if score_type == "Train" else '-'

#                 ax.plot(
#                     subset["Train Set Size"],
#                     subset["Score"],
#                     label=None,
#                     marker=marker,
#                     markersize=8,
#                     linestyle=linestyle,
#                     linewidth=2.5,
#                     color=color
#                 )
#                 ax.fill_between(
#                     subset["Train Set Size"],
#                     subset["Score"] - subset["Std"],
#                     subset["Score"] + subset["Std"],
#                     alpha=0.2,
#                     color=color
#                 )

#         ax.set_xticks(np.arange(0, 251, 50))
#         ax.set_ylim(0, max_score)
#         ax.set_yticks(score_yticks)
#         ax.tick_params(axis='x', labelsize=fontsize)
#         ax.tick_params(axis='y', labelsize=fontsize)
#         ax.set_xlabel("Training Set Size", fontsize=fontsize + 2, fontweight='bold')

#         for spine in ['top', 'right', 'bottom', 'left']:
#             ax.spines[spine].set_visible(True)

#     g.map_dataframe(plot_accuracy)

#     for i, ax in enumerate(g.axes.flat):
#         if i == 0:
#             ax.set_ylabel(metric.upper(), fontsize=fontsize + 2, fontweight='bold')
#         else:
#             ax.set_ylabel("")
#             ax.set_yticklabels([])

#         ax.set_title(g.col_names[i], fontsize=fontsize + 4)

#     # Combined Legend
#     legend_elements = [
#         Line2D([0], [0], color=color_map['CO_'], marker='o', linestyle='--', lw=3, markersize=10, label="Train OOD"),
#         Line2D([0], [0], color=color_map['CO_'], marker='^', linestyle='-', lw=3, markersize=12, label="Test OOD"),
#         Line2D([0], [0], color=color_map['IID_'], marker='o', linestyle='--', markersize=10, lw=3, label="Train IID"),
#         Line2D([0], [0], color=color_map['IID_'], marker='^', linestyle='-', markersize=12, lw=3, label="Test IID"),
#     ]

#     if title:
#         g.fig.suptitle(title, fontsize=fontsize + 10, fontweight='bold', y=1.12)

#     plt.figlegend(
#         handles=legend_elements,
#         loc='upper center',
#         bbox_to_anchor=(0.5, 1.12),
#         ncol=4,
#         fontsize=fontsize,
#         frameon=True
#     )

#     plt.tight_layout(rect=[0, 0, 1, 0.98])

#     if folder:
#         save_img_path(folder, f"accuracy comparison ({file_name}).png")

#     plt.show()
#     plt.close()





# comparison_of_features_lc = {
#     '(Mordred-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_hypOFF_Standard_lc':'Xn + + Mordred ',
#     '(Mordred-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_RF_hypOFF_Standard_lc':'Mordred+continuous+aging',
#     '(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_hypOFF_Standard_lc':'continuous',
#     '(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_RF_hypOFF_Standard_lc':'continuous+aging',
# }

def get_comparison_of_features(model: str, suffix: str):
    return {
        # f'(concentration-temperature-solvent dP-solvent dD-solvent dH)_{model}{suffix}': 'solvent_properties + solvent_HSPs',
        # f'(concentration-temperature-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_{model}{suffix}': 'solvent_properties + solvent_HSPs + environmental.thermal history',
        # f'(light exposure-aging time-aging temperature-prep temperature-prep time)_{model}{suffix}': 'environmental.thermal history',
        # f'(Xn)_{model}{suffix}': 'Xn',
        # f'(Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize',
        # f'(Xn-Mw-PDI-polymer dP-polymer dD-polymer dH)_{model}{suffix}': 'Xn + polysize + polymer HSPs',
        f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_{model}{suffix}': 'combination of all',
        # f'(Mordred-Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize + Mordred',
        # f'(MACCS-Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize + MACCS',
        # f'(ECFP3.count.512-Xn-Mw-PDI)_{model}{suffix}': 'Xn + polysize + ECFP6.count.512',
    }

# Example usage:


if __name__ == "__main__":
    HERE: Path = Path(__file__).resolve().parent
    results_path = HERE.parent.parent / 'results'/ 'OOD_target_log Rg (nm)'
    cluster_list = [

                    # 'HBD3 MACCS cluster',
                    # 'substructure cluster',
                    # 'KM4 ECFP6_Count_512bit cluster',	
                    # 'KM3 Mordred cluster',
                    # 'KM4 polymer_solvent HSP cluster',
                    # 'KM4 Mordred_Polysize cluster',
                    'Polymers cluster'
                    ]



    for cluster in cluster_list:
        # scores_folder_path = results_path / cluster / 'Trimer_scaler'
        # for fp in ['MACCS', 'Mordred']:
        #     all_score_eq_training_size = []
            # for model in ['RF']:

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

        # Plot uncertenty + score in learning curve for comparison of features ##############
        # model = 'RF'
        # comparison_of_features_lc = get_comparison_of_features(model, "_hypOFF_Standard_lc")

        # all_score_eq_training_size = []
        # uncertenty_method = 'AMA'
        # all_model_predictions = {}
        # models = ['RF', 'XGBR']

        # for model in models:
        #     comparison_of_features_lc = get_comparison_of_features(model, "_hypOFF_Standard_lc")

        #     for file, file_discription in comparison_of_features_lc.items():

        #         if any(keyword in file for keyword in ['Mordred', 'MACCS', 'ECFP3.count.512']):
        #             scores_folder_path = results_path / cluster / 'Trimer_scaler'
        #         else:
        #             scores_folder_path = results_path / cluster / 'scaler'
                
        #         score_file_lc = ensure_long_path(scores_folder_path / f'{file}_scores.json')
        #         predictions_file_lc = ensure_long_path(scores_folder_path / f'{file}_predictions.json')

        #         if not os.path.exists(score_file_lc) or not os.path.exists(predictions_file_lc):
        #             print(f"File not found: {file}")
        #             continue  

        #         with open(score_file_lc, "r") as f:
        #             scores_lc = json.load(f)
        #         with open(predictions_file_lc, "r") as s:
        #             predictions_lc = json.load(s)


        #         all_model_predictions[model] = predictions_lc
        #         saving_folder_lc_score_of_features = results_path / cluster / f'comparison of features learning curve'            
        #         plot_ood_learning_accuracy_uncertainty(scores_lc, predictions_lc, metric="rmse",
        #                                                 folder=saving_folder_lc_score_of_features,
        #                                                 file_name=f'{file_discription}_{uncertenty_method}_{model}', uncertenty_method=uncertenty_method,
        #                                                 title=file_discription
        #                                                 )
        #         print("Save learning curve scores and uncertainty")


                # plot_ood_learning_uncertainty(prediction=predictions_lc, folder= saving_folder_lc_score_of_features,
                #                             file_name=f'{file_discription}_{model}')
                
                # print("Save learning curve uncertainties")


        # Plot ood_learning_uncertainty and score new

        # all_model_predictions = {}
        # all_model_scores = {}

        # models = ['RF', 'XGBR']
        # for model in models:
        #     scores_folder_path = results_path / cluster / 'scaler'
        #     score_file_lc = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_{model}_hypOFF_Standard_lc_scores.json'
        #     predictions_file_lc = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_{model}_hypOFF_Standard_lc_predictions.json'
        #     predictions_file_lc = ensure_long_path(predictions_file_lc)
        #     score_file_lc = ensure_long_path(score_file_lc)
        #     if not os.path.exists(score_file_lc) or not os.path.exists(predictions_file_lc):
        #         print(f"File not found: {predictions_file_lc}")
        #         continue  

        #     with open(score_file_lc, "r") as f:
        #         scores_lc = json.load(f)
        #     with open(predictions_file_lc, "r") as s:
        #         predictions_lc = json.load(s)

        #     all_model_predictions[model] = predictions_lc
        #     all_model_scores[model] = scores_lc
        # saving_folder = results_path / cluster / 'comparison of models learning curve (score and uncertainty)'
        # plot_ood_learning_uncertainty(
        #     model_predictions=all_model_predictions,
        #     folder=saving_folder,
        #     file_name=f"combination of all features",
        #     fontsize=20,
        #     uncertainty_methods=("Spearman R", "AMA"),
        #     # title=f"Uncertainty Comparison for {file_discription}"
        # )

        # print("Saved model uncertainty comparison plots.")


        # plot_ood_learning_accuracy_only(
        #     all_scores=all_model_scores,
        #     metric="rmse",
        #     folder=saving_folder,
        #     file_name="accuracy only combination of all features_RF",
        #     # title="Model Accuracy Comparison",
        #     fontsize=18
        # )

        # print("Saved model accuracy-only learning curve plot.")



            # Plot OOD vs IID barplot at the same training size for comparison of features
            
        #     _, ood_iid_eq_tarining_size_df = process_learning_curve_scores(scores_lc, metric="rmse")
        #     ood_iid_eq_tarining_size_df = ood_iid_eq_tarining_size_df[ood_iid_eq_tarining_size_df["Score Type"] == "Test"]
        #     ood_iid_eq_tarining_size_df['Model'] = file_discription
        #     all_score_eq_training_size.append(ood_iid_eq_tarining_size_df)
        # all_score_eq_training_size: pd.DataFrame = pd.concat(all_score_eq_training_size, ignore_index=True)
        # saving_folder = results_path / cluster / f'OOD-IID bar plot at equal training set (comparison of features)'
        # plot_bar_ood_iid(all_score_eq_training_size, 'rmse', 
        #                  saving_folder, file_name=f'rmse_{model}_comparison of feature',
        #                      text_size=16, figsize=(15, 8), ncol=3)

        # print("save OOD vs IID bar plot at equal training size for comparison of features")

        # plot OOD vs IID barplot at the same training size 
        accuracy_metric = "rmse"
        all_score_eq_training_size = []
        for model in ['XGBR', 'RF']:

                scores_folder_path = results_path / cluster / 'scaler'
                scores_lc = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time)_{model}_hypOFF_Standard_lc_scores.json'
                scores_lc = ensure_long_path(scores_lc)
                if not os.path.exists(scores_lc):
                    print(f"File not found: {scores_lc}")
                    continue  
                with open(scores_lc, "r") as f:
                    scores = json.load(f)
                _, ood_iid_eq_tarining_size_df = process_learning_curve_scores(scores, metric=accuracy_metric)
                ood_iid_eq_tarining_size_df = ood_iid_eq_tarining_size_df[ood_iid_eq_tarining_size_df["Score Type"] == "Test"]
                ood_iid_eq_tarining_size_df['Model'] = model

                all_score_eq_training_size.append(ood_iid_eq_tarining_size_df)
        all_score_eq_training_size:pd.DataFrame = pd.concat(all_score_eq_training_size, ignore_index=True)
        saving_folder = scores_folder_path/  f'OOD-IID bar plot(equal training set)'
        
        figsize = (16, 10) if cluster == 'Polymers cluster' else (8, 6)
        plot_bar_ood_iid(all_score_eq_training_size, accuracy_metric, saving_folder, file_name=f'metric-{accuracy_metric}',
                            text_size=22,figsize=figsize, ncol=2)
        print("save OOD vs IID bar plot at equal training size")

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







