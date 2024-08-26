import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits
import json
import numpy as np
from clean_dataset import open_json
from argparse import ArgumentParser
from data_handling import save_results
import sys
sys.path.append("../cleaning")


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training_dataset"/ "structure_wo_block_cp_scaler_dataset.pkl"
oligo_dir: Path = DATASETS/ "raw"/"pu_columns_used.json"

oligomer_list =open_json(oligo_dir)
w_data = pd.read_pickle(training_df_dir)
edited_oligomer_list = [" ".join(x.split()[:-1]) for x in oligomer_list]





def main_MACCS_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    oligomer_representation: str
) -> None:
    representation: str = "MACCS"
    structural_features: list[str] = [f"{oligomer_representation}_{representation}"]
    unroll_single_feat = {"representation": representation,
                          "oligomer_representation":oligomer_representation,
                          "col_names": structural_features}

    scores, predictions  = train_regressor(
                            dataset=dataset,
                            features_impute=None,
                            special_impute=None,
                            representation=representation,
                            structural_features=structural_features,
                            unroll=unroll_single_feat,
                            numerical_feats=None,
                            target_features=target_features,
                            regressor_type=regressor_type,
                            transform_type=transform_type,
                            hyperparameter_optimization=hyperparameter_optimization,
                            imputer=None
                        )