import json
from itertools import product
from pathlib import Path
from typing import List
import os 
# import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from matplotlib import rc


HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE.parent.parent/ 'results'

targer_dir: Path = Path(RESULTS/'target_Lp')


# with open(filters, "r") as f:
#     FILTERS: dict = json.load(f)

# score_bounds: dict[str, int] = {"r": 1, "r2": 1, "mae": 7.5, "rmse": 7.5}
var_titles: dict[str, str] = {"stdev": "Standard Deviation", "stderr": "Standard Error"}







file = Path(r'C:\Users\sdehgha2\Desktop\PhD code\pls-dataset-project\PLS-Dataset\results\target_Lp\Trimer\(ECFP8)_count_1024_RF_scores.json')
# print(file.name)

def creat_polymer_unit_fingerprint() -> None:

    pattern: str = "*_scores.json"

    for representation in os.listdir(targer_dir):
        
        if representation != 'test' and 'scaler' not in representation:
            representation_dir = os.path.join(targer_dir, representation)
            score_files: list[Path] = list(Path(representation_dir).rglob(pattern))
            
            for file_path in score_files:
                print(file)
                

                features = file.parent.name.split("_")[-1]

                with open(file, "r") as f:
                    data = json.load(f)

    
    
    # for parent in parent_dir_labels:
    #     p = root_dir / parent
    #     for feats, model, avg, std in get_results_from_file(p, score, var, **kwargs):
    #         if model is None:
    #             continue
    #         if model not in annotations.index or feats not in annotations.columns:
    #             continue
    #         avg_scores.at[model, feats] = avg
    #         std_scores.at[model, feats] = std

def get_results_from_file(
    file_path: Path,
    score: str,
    var: str,
    # impute: bool = False,
) -> tuple[float, float]:
    """
    Args:
        root_dir: Root directory containing all results
        representation: Representation for which to get scores
        model: Model for which to get scores.
        score: Score to plot
        var: Variance to plot

    Returns:
        Average and variance of score
    """

    if not file_path.exists():
        features, model = None, None
        avg, std = np.nan, np.nan
    else:
        if "imputer" in file_path.name:
            pass

            # features = file.name.split("_")[-1]
        # features =  if impute else file.parent.name.split("_")[-1]
        else:

            model:str = file_path.name.split("_")[-2]
            features:str = "-".join(file_path.name.split("_")[:-2])
            # above can be fingerprints
        with open(file_path, "r") as f:
            data = json.load(f)


        avg = data[f"{score}_avg"]
        if var == "stdev":
            std = data[f"{score}_stdev"]
        elif var == "stderr":
            std = data[f"{score}_stderr"]
        else:
            raise ValueError(f"Unknown variance type: {var}")
        # se = data[f"{score}_stderr"]

        # avg: float = np.nan if abs(avg) > score_bounds[score] else avg
        # std: float = np.nan if abs(std) > score_bounds[score] else std
        # se: float = np.nan if abs(se) > score_bounds[score] else se

        if score in ["mae", "rmse"]:
            avg, std = abs(avg), abs(std)
        return features, model, avg, std

feat, model, av, std = get_results_from_file(file,score='r2', var='stdev')

print(feat,model, av ,std)

def generate_annotations(num: float) -> str:
    """
    Args:
        num: Number to annotate

    Returns:
        String to annotate heatmap
    """
    if isinstance(num, float) and not np.isnan(num):
        num_txt: str = f"{round(num, 2)}"
    else:
        num_txt = "NaN"
    return num_txt






def _create_heatmap(
    root_dir: Path,
    score: str,
    var: str,
    x_labels: list[str],
    y_labels: list[str],
    parent_dir_labels: list[str],
    figsize: tuple[int, int],
    fig_title: str,
    x_title: str,
    y_title: str,
    fname: str,
    vmin: float = None,
    vmax: float = None,
    **kwargs,
) -> None:
    """
    Args:
        root_dir: Root directory containing all results
        score: Score to plot
        var: Variance to plot
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        figsize: Figure size
        fig_title: Figure title
        x_title: X-axis title
        y_title: Y-axis title
        fname: Filename to save figure
    """
    # avg_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
    # std_scores: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
    # annotations: pd.DataFrame = pd.DataFrame(columns=x_labels, index=y_labels)
    # for parent, model in product(parent_dir_labels, y_labels):

    
    
    
    for parent in parent_dir_labels:
        p = root_dir / parent
        for feats, model, avg, std in get_results_from_file(p, score, var, **kwargs):
            if model is None:
                continue
            if model not in annotations.index or feats not in annotations.columns:
                continue
            avg_scores.at[model, feats] = avg
            std_scores.at[model, feats] = std
    for x, y in product(x_labels, y_labels):
        avg: float = avg_scores.loc[y, x]
        std: float = std_scores.loc[y, x]
        avg_txt: str = generate_annotations(avg)
        std_txt: str = generate_annotations(std)
        annotations.loc[y, x] = f"{avg_txt}\n±{std_txt}"

    avg_scores = avg_scores.astype(float)
    annotations = annotations.astype(str)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    palette: str = "viridis" if score in ["r", "r2"] else "viridis_r"
    # # palette: str = "cmc.batlow" if score in ["r", "r2"] else "cmc.batlow_r"
    # palette: str = "cmc.batlow_r" if score in ["r", "r2"] else "cmc.batlow"
    custom_cmap = sns.color_palette(palette, as_cmap=True)
    custom_cmap.set_bad(color="lightgray")
    hmap = sns.heatmap(
        avg_scores,
        annot=annotations,
        fmt="",
        cmap=custom_cmap,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        mask=avg_scores.isnull(),
        annot_kws={"fontsize": 12},
    )
    # Set axis labels and tick labels
    ax.set_xticks(np.arange(len(avg_scores.columns)) + 0.5)
    ax.set_yticks(np.arange(len(avg_scores.index)) + 0.5)
    x_tick_labels: list[str] = [col for col in avg_scores.columns]
    try:  # handles models as y-axis
        y_tick_labels: list[str] = [model_abbrev_to_full[x] for x in avg_scores.index]
    except:  # handles targets as y-axis
        y_tick_labels: list[str] = [target_abbrev_to_full[x] for x in avg_scores.index]
    if kwargs.get("multioutput"):
        ax.set_xticklabels(x_tick_labels, rotation=0)
    else:
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_tick_labels, rotation=0, ha="right")

    # Set plot and axis titles
    plt.title(fig_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    # Set colorbar title
    score_txt: str = "$R^2$" if score == "r2" else score
    cbar = hmap.collections[0].colorbar
    cbar.set_label(
        f"Average {score_txt.upper()} ± {var_titles[var]}", rotation=270, labelpad=20
    )

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(root_dir / f"{fname}.png", dpi=600)

    # Show the heatmap
    # plt.show()
    plt.close()


    




def create_grid_search_heatmap(root_dir: Path, score: str, var: str) -> None:
    """
    Args:
        root_dir: Root directory containing all results
        score: Score to plot
        var: Variance to plot
    """
    x_labels: List[str] = [
        "fabrication only",
        "OHE",
        "material properties",
        "SMILES",
        "SELFIES",
        "ECFP",
        "mordred",
        "graph embeddings",
        "GNN",
    ]
    y_labels: List[str] = [
        "MLR",
        "KNN",
        "SVR",
        "KRR",
        "GP",
        "RF",
        "HGB",
        "XGB",
        "NGB",
        "NN",
        # "ANN",
        "GNN",
    ][::-1]
    parent_dir_labels: list[str] = [f"features_{x}" for x in x_labels]

    target: str = ", ".join(root_dir.name.split("_")[1:])
    score_txt: str = "$R^2$" if score == "r2" else score.upper()

    _create_heatmap(
        root_dir,
        score,
        var,
        x_labels=x_labels,
        y_labels=y_labels,
        parent_dir_labels=parent_dir_labels,
        figsize=(12, 8),
        fig_title=f"Average {score_txt} Scores for Models Predicting {target}",
        x_title="Structural Representations",
        y_title="Regression Models",
        fname=f"model-representation search heatmap_{score}",
    )
