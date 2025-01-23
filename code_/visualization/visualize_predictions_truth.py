import json
from math import ceil
from pathlib import Path
import os 
import numpy as np
# import cmcrameri.cm as cmc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rc


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS: Path = HERE.parent.parent/ 'results'
target_list = ['target_Rg']

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"],
              "size": 12
              })


# def get_predictions(directory: Path, ground_truth_file: Path) -> None:
#     true_values: pd.Series = pd.read_csv(ground_truth_file)["calculated PCE (%)"] 
#     for pred_file in directory.rglob("*_predictions.csv"):
#         # print(pred_file)
#         model_name: str = pred_file.stem.split("_")[0]
#         r2_avg, r2_stderr = get_scores(pred_file.parent, model_name)

#         make_predictions_plot(true_values, pred_file, r2_avg, r2_stderr)


def get_file_info(file_path:Path)-> tuple[str, str]:
    # return file name and polymer representation
    return file_path.stem , file_path.parent.name

    # result_must_be target dir
def get_prediction_plot(results_directory: Path, ground_truth_file: Path, target:str)->None:
    # TODO: it should drop the values
    w_data = pd.read_pickle(ground_truth_file)
    true_values: pd.DataFrame = w_data[target].dropna()
    # true_values: pd.Series = pd.read_csv(ground_truth_file)["calculated PCE (%)"]
    
    pattern: str = "*_predictions.csv"
    for representation in os.listdir(results_directory):
        score_files = []
            
        representation_dir = os.path.join(results_directory, representation)
        score_files: list[Path] = list(Path(representation_dir).rglob(pattern))
        
        for pred_path in score_files:
            file_name, polymer_representation = get_file_info(pred_path)
            print(file_name)
            r2_avg, r2_stderr = get_scores(pred_path)
            draw_predictions_plot(true_values,
                                    pred_path,
                                    r2_avg,
                                    r2_stderr,
                                    results_directory,
                                    polymer_representation,
                                    file_name)





def get_scores(dir: Path,peak_number:int) -> tuple[float, float]:
    score_path = dir.with_name(dir.stem.replace("_prediction", "_score") + ".json")
    with open(score_path, "r") as f:
        scores: dict = json.load(f)
        
    avg = scores["r2_avg"]
    std = scores["r2_stdev"]
    avg = avg[peak_number] if isinstance(avg, list) else avg
    std = std[peak_number] if isinstance(std, list) else std
    r2_avg = round(avg, 2)
    r2_stderr = round(std, 2)
    return r2_avg, r2_stderr


# def draw_predictions_plot(
#                         #    true_values: pd.Series,
#                         target:str,
#                         predictions: Path,
#                         # r2_avg: float,
#                         # r2_stderr: float,
#                         root_dir:Path,
#                         poly_representation_name:str,
#                         file_name,
#                            ) -> None:
#     # Load the data from CSV files
    
#     predicted_values = pd.read_csv(predictions)
#     seeds = predicted_values.drop(target, axis=1).columns 

#     # There are 7 columns in predicted_values, each corresponding to a different seed
#     # Create a Series consisting of the ground truth values repeated 7 times
#     true_values_ext = pd.concat([predicted_values[target]] * len(seeds), ignore_index=True)
#     # Create a Series consisting of the predicted values, with the column names as the index
#     predicted_values_ext = pd.concat([predicted_values[col] for col in seeds], axis=0, ignore_index=True)

#     ext_comb_df = pd.concat([true_values_ext, predicted_values_ext], axis=1)
#     print(ext_comb_df)
#     log_ext_comb_df = np.log10(ext_comb_df+1)

#     # print(ext_comb_df)
#     # Create the hex-binned plot with value distributions for all y-axis columns

#     g = sns.jointplot(data=log_ext_comb_df, x=target, y=0,
#                       kind="hex",
#                     #   cmap="viridis",
#                       # joint_kws={"gridsize": 50, "cmap": "Blues"},
#                     #   joint_kws={"gridsize": (150,45)},
#                       marginal_kws={"bins": 25},
#                       )
#     ax_max = ceil(max(log_ext_comb_df.max()))
#     # print(ax_max)
#     g.ax_joint.plot([0, ax_max], [0, ax_max], ls="--", c=".3")
#     # g.ax_joint.annotate(f"$R^2$ = {r2_avg} ± {r2_stderr}",
#     #                     # xy=(0.1, 0.9), xycoords='axes fraction',
#     #                     # ha='left', va='center',
#     #                     # bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}
#     #                     )
#     # TODO:
#     #  kwargs: linewidth=0.2, edgecolor='white',  mincnt=1
#     # plt.text(0.95, 0.05, f"$R^2$ = {r2_avg} ± {r2_stderr}",
#     #          horizontalalignment='right',
#     #          verticalalignment='bottom',
#     #          transform=g.ax_joint.transAxes,
#     #          )
#     # g.plot_marginals(sns.kdeplot, color="blue")
#     # Set plot limits to (0, 15) for both axes
#     g.set_axis_labels(f"log True {target} ", f"log Predicted {target}")
#     g.ax_joint.set_xlim(0, ax_max)
#     g.ax_joint.set_ylim(0, ax_max)
#     # plt.tight_layout()
#     # g.ax_joint.set_xscale("log")
#     # g.ax_joint.set_yscale("log")
#     # plt.show()


#     visualization_folder_path =  root_dir/"parity plot"/poly_representation_name
#     os.makedirs(visualization_folder_path, exist_ok=True)
#     saving_path = visualization_folder_path/ f"{file_name}.png"

#     # output: Path = predictions.parent / f"{predictions.stem}_plot.png"
#     # plt.savefig(output, dpi=600)
#     try:
#         g.savefig(saving_path, dpi=600)
#         print(f"Saved {saving_path}")
#     except Exception as e:
#         print(f"Failed to save {saving_path} due to {e}")

#     # plt.show() (if needed)
#     plt.close()


def draw_predictions_plot(
                        target:str,
                        predictions: Path,
                        # r2_avg: float,
                        # r2_stderr: float,
                        root_dir:Path,
                        poly_representation_name:str,
                        file_name,
                        peak_num,
                           ) -> None:
    # Load the data from CSV files
    r2_avg, r2_stderr = get_scores(predictions,peak_number=peak_num)
    predicted_values = pd.read_csv(predictions)
    seeds = predicted_values.drop(target, axis=1).columns 
    restored_df = pd.DataFrame({
    col: predicted_values[col].values.reshape(-1, 3).tolist()  # Reshape each column
    for col in predicted_values.columns
    })
    restored_df[target] = restored_df[target].apply(lambda x: eval(x)[peak_num] if isinstance(x, str) else x[peak_num])
    true_values_ext = pd.concat([restored_df[target]] * len(seeds), ignore_index=True)

    # There are 7 columns in predicted_values, each corresponding to a different seed
    # Create a Series consisting of the ground truth values repeated 7 times
    # Create a Series consisting of the predicted values, with the column names as the index
    predicted_values_ext = pd.concat([restored_df[col].apply(lambda x: eval(x)[peak_num] if isinstance(x, str) else x[peak_num]) for col in seeds], axis=0, ignore_index=True)

    ext_comb_df = pd.concat([true_values_ext, predicted_values_ext], axis=1)
    # print(ext_comb_df)
    log_ext_comb_df = np.log10(ext_comb_df+1)

    # print(ext_comb_df)
    # Create the hex-binned plot with value distributions for all y-axis columns

    g = sns.jointplot(data=log_ext_comb_df, x=target, y=0,
                      kind="hex",
                    #   cmap="viridis",
                      # joint_kws={"gridsize": 50, "cmap": "Blues"},
                    #   joint_kws={"gridsize": (150,45)},
                      marginal_kws={"bins": 25},
                    #   gridsize=150,  # Reduce gridsize for larger hexagons
                      )
    ax_max = ceil(max(log_ext_comb_df.max()))
    ax_min = ceil(min(log_ext_comb_df.min()))
    g.ax_joint.plot([0, ax_max], [0, ax_max], ls="--", c=".3")
    g.ax_joint.annotate(f"$R^2$ = {r2_avg} ± {r2_stderr}",
                        xy=(0.1, 0.9), xycoords='axes fraction',
                        ha='left', va='center',
                        bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}
                        )
    #  kwargs: linewidth=0.2, edgecolor='white',  mincnt=1
    # plt.text(0.95, 0.05, f"$R^2$ = {r2_avg} ± {r2_stderr}",
    #          horizontalalignment='right',
    #          verticalalignment='bottom',
    #          transform=g.ax_joint.transAxes,
    #          )
    # g.plot_marginals(sns.kdeplot, color="blue")
    # Set plot limits to (0, 15) for both axes
    g.set_axis_labels(f"log True {target} ", f"log Predicted {target}")
    g.ax_joint.set_xlim(ax_min, ax_max)
    g.ax_joint.set_ylim(ax_min, ax_max)
    # plt.tight_layout()
    # g.ax_joint.set_xscale("log")
    # g.ax_joint.set_yscale("log")
    # plt.show()


    visualization_folder_path =  root_dir/"parity plot"/poly_representation_name
    os.makedirs(visualization_folder_path, exist_ok=True)
    peak_name = peak_num+1
    saving_path = visualization_folder_path/ f"{file_name}_{peak_name}.png"

    # output: Path = predictions.parent / f"{predictions.stem}_plot.png"
    # plt.savefig(output, dpi=600)
    try:
        g.savefig(saving_path, dpi=600)
        print(f"Saved {saving_path}")
    except Exception as e:
        print(f"Failed to save {saving_path} due to {e}")

    # plt.show() (if needed)
    plt.close()





if __name__ == "__main__":
    # predictions = HERE.parent.parent / "results" / "target_PCE" / "features_ECFP" / "RF_predictions.csv"
    # make_predictions_plot(predictions, 0.87, 0.02)
    # dataset_ground_truth_csv = DATASETS / "Min_2020_n558" / "cleaned_dataset.csv"
    # ground_truth_Hutchison_csv = DATASETS / "Hutchison_2023_n1001" / "Hutchison_filtered_dataset_pipeline.csv"
    # ground_truth_Saeki_csv = DATASETS / "Saeki_2022_n1318" / "Saeki_corrected_pipeline.csv"

    # training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.pkl"
    # for target_folder in target_list:
    #     get_prediction_plot(RESULTS/target_folder, training_df_dir,"Rg1 (nm)")

    # for result_dir, ground_truth_csv in zip(["results_Hutchison", "results_Saeki"], [ground_truth_Hutchison_csv, ground_truth_Saeki_csv]):
    #     pce_results = ROOT / result_dir
    #     get_predictions(pce_results, ground_truth_csv)
    target_to_an = 'target_multimodal Rh'
    file_n = "(PDI-Mw-concentration-temperature-solvent dP-solvent dD-solvent dH)_XGBR_mean_Robust Scaler_predictions.csv"
    poly_representation_name = 'test_without_log'
    truth_val_file:Path = RESULTS/target_to_an/poly_representation_name/file_n
    fname,_ = get_file_info(truth_val_file)
    # for i in [0,1,2]:
    draw_predictions_plot(
        target='multimodal Rh',
        predictions=truth_val_file,
        # r2_avg: float,
        # r2_stderr: float,
        root_dir=RESULTS/target_to_an,
        poly_representation_name=poly_representation_name,
        peak_num=2,
        file_name=fname,
    )