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

target_dir: Path = Path(RESULTS/'target_Lp')


# with open(filters, "r") as f:
#     FILTERS: dict = json.load(f)

# score_bounds: dict[str, int] = {"r": 1, "r2": 1, "mae": 7.5, "rmse": 7.5}
var_titles: dict[str, str] = {"stdev": "Standard Deviation", "stderr": "Standard Error"}







file = Path(r'C:\Users\sdehgha2\Desktop\PhD code\pls-dataset-project\PLS-Dataset\results\target_Lp\Trimer\(ECFP8)_count_1024_RF_scores.json')
# print(file.name)


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
        # for just scaler features
        if '(numerical)'== file_path.name.split("_")[0]:
            model:str = file_path.name.split("_")[-3] 
            features:str = file_path.name.split("_")[0]
        else:
            features:str = "-".join(file_path.name.split("_")[:-2])
            # for mixure of scaler and fingerprint
            if "imputer" in file_path.name:
                model:str = file_path.name.split("_")[-3] 
            # for fingerprints only
            else:   
                model:str = file_path.name.split("_")[-2]
       
        with open(file_path, "r") as f:
            data = json.load(f)


        avg = data[f"{score}_avg"]
        if var == "stdev":
            std = data[f"{score}_stdev"]
        elif var == "stderr":
            std = data[f"{score}_stderr"]
        else:
            raise ValueError(f"Unknown variance type: {var}")

        if score in ["mae", "rmse"]:
            avg, std = abs(avg), abs(std)
        return features, model, avg, std


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
    avg_scores:pd.DataFrame,
    annotations:pd.DataFrame,
    # x_labels: list[str],
    # y_labels: list[str],
    # parent_dir_labels: list[str],
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
    # try:  # handles models as y-axis
    # except:  # handles targets as y-axis
    #     y_tick_labels: list[str] = [target_abbrev_to_full[x] for x in avg_scores.index]
    y_tick_labels: list[str] = avg_scores.index.to_list()

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
    visualization_folder_path =  root_dir/"heatmap"
    os.makedirs(visualization_folder_path, exist_ok=True)    
    plt.tight_layout()
    plt.savefig(visualization_folder_path / f"{fname}.png", dpi=600)

    # Show the heatmap
    # plt.show()
    plt.close()



def creat_result_df(target_dir: Path,
                    score: str,
                    var: str,
                    data_type:str
) -> tuple[pd.DataFrame,pd.DataFrame]:
    
    avg_scores: pd.DataFrame = pd.DataFrame()
    std_scores: pd.DataFrame = pd.DataFrame()
    annotations: pd.DataFrame = pd.DataFrame()
    models = set()
    pattern: str = "*_scores.json"
    for representation in os.listdir(target_dir):
        score_files = []
        if data_type == 'structural':
            
            if representation != 'test' and 'scaler' not in representation:
                filterd_rep = representation
                representation_dir = os.path.join(target_dir, representation)
                score_files: list[Path] = list(Path(representation_dir).rglob(pattern))
        elif data_type == 'scaler':
        
            if  'scaler' == representation:
                filterd_rep = representation
                representation_dir = os.path.join(target_dir, representation)
                score_files: list[Path] = list(Path(representation_dir).rglob(pattern))
        
        elif data_type=='structural_scaler':
            
            if  'scaler' in representation and 'scaler' != representation:
                filterd_rep = representation
                representation_dir = os.path.join(target_dir, representation)
                score_files: list[Path] = list(Path(representation_dir).rglob(pattern))

        for file_path in score_files:
            
            feats, model, av , std = get_results_from_file(file_path=file_path, score=score, var=var)                
            models.add(model)
            # for scaler features only
            if data_type=='scaler':
                if feats not in avg_scores.columns:

                    avg_scores.loc[model,feats] = av
                    std_scores.loc[model,feats] = std
                else:
                    avg_scores.at[model,feats] = av
                    std_scores.at[model,feats] = std
            # for mixtures and fingerprints 
            else:
                if feats not in avg_scores.columns:

                    avg_scores.loc[filterd_rep,feats] = av
                    std_scores.loc[filterd_rep,feats] = std
                else:
                    avg_scores.at[filterd_rep,feats] = av
                    std_scores.at[filterd_rep,feats] = std
        


    for x, y in product(avg_scores.columns.to_list(), avg_scores.index.to_list()):
        avg: float = avg_scores.loc[y, x]
        std: float = std_scores.loc[y, x]
        avg_txt: str = generate_annotations(avg)
        std_txt: str = generate_annotations(std)
        annotations.loc[y, x] = f"{avg_txt}\n±{std_txt}"

    avg_scores = avg_scores.astype(float)
    annotations = annotations.astype(str)


    return avg_scores, annotations, list(models)



def create_structural_result(target_dir:Path,
                               score:str,
                               var:str,
                               data_type:str
                               ) -> None:
    ave, anot, model = creat_result_df(target_dir=target_dir,score=score, var=var,data_type=data_type)
    model_in_title:str =  ",".join(model)
    traget: str = 'Lp (nm)'
    score_txt: str = "$R^2$" if score == "r2" else score.upper()
    _create_heatmap(root_dir=target_dir,
                    score=score,
                    var=var,
                    avg_scores=ave,
                    annotations=anot,
                    figsize=(12, 8),
                    fig_title=f"Average {score_txt} Scores for Fingerprint Predicting {traget} using {model_in_title} model(s)",
                    x_title="Fingerprint Representations",
                    y_title="Polymer Unit Representation",
                    fname=f"PolymerRepresentation vs Fingerprint search heatmap_{score}")


scores_list: list = {"r", "r2", "mae", "rmse"}
var_titles: dict[str, str] = {"stdev": "Standard Deviation", "stderr": "Standard Error"}
# for i in scores_list:
#     create_structural_result(target_dir=target_dir,score=i,var='stdev',data_type='structural')



# feat, model, av, std = get_results_from_file(file,score='r2', var='stdev')

# print(feat,model, av ,std)

# def create_structural_scaler_result() -> None:




def create_structural_scaler_result(target_dir:Path,
                                    score:str,
                                    var:str,
                                    data_type:str
                                    ) -> None:

    ave, anot, model = creat_result_df(target_dir=target_dir,score=score, var=var,data_type=data_type)
    model_in_title:str =  ",".join(model)
    traget: str = 'Lp (nm)'
    score_txt: str = "$R^2$" if score == "r2" else score.upper()
    _create_heatmap(root_dir=target_dir,
                    score=score,
                    var=var,
                    avg_scores=ave,
                    annotations=anot,
                    figsize=(12, 8),
                    fig_title=f"Average {score_txt} Scores for Fingerprint-numerical Predicting {traget} using {model_in_title} model",
                    x_title="Fingerprint-numerical Representations",
                    y_title="Polymer Unit Representation",
                    fname=f"PolymerRepresentation vs (Fingerprint-numerical) search heatmap_{score}")
    


# for i in scores_list:
#     create_structural_scaler_result(target_dir=target_dir,score=i,var='stdev',data_type='structural_scaler')


def create_scaler_result(target_dir:Path,
                        score:str,
                        var:str,
                        data_type:str
                        )->None:

    ave, anot, model = creat_result_df(target_dir=target_dir,score=score, var=var,data_type=data_type)
    model_in_title:str =  ",".join(model)
    traget: str = 'Lp (nm)'
    score_txt: str = "$R^2$" if score == "r2" else score.upper()
    _create_heatmap(root_dir=target_dir,
                    score=score,
                    var=var,
                    avg_scores=ave,
                    annotations=anot,
                    figsize=(12, 8),
                    fig_title=f"Average {score_txt} Scores for numerical Predicting {traget} using {model_in_title} model",
                    x_title="numerical Representations",
                    y_title="Regression Models",
                    fname=f"Regression Models vs numerical features search heatmap_{score}")
    

for i in scores_list:
    create_scaler_result(target_dir=target_dir,score=i,var='stdev',data_type='scaler')