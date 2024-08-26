import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits
import json
import numpy as np
import sys
sys.path.append("../cleaning")
from clean_dataset import open_json
from argparse import ArgumentParser
from data_handling import save_results


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training_dataset"/ "structure_wo_block_cp_scaler_dataset.pkl"
oligo_dir: Path = DATASETS/ "raw"/"pu_columns_used.json"

oligomer_list =open_json(oligo_dir)
w_data = pd.read_pickle(training_df_dir)
edited_oligomer_list = [" ".join(x.split()[:-1]) for x in oligomer_list]



TEST = False




def main_numerical_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:


    columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    special_column: str = "Mw (g/mol)"
    numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]

    imputer = "mean"
    scores, predictions  = train_regressor(
                            dataset=dataset,
                            features_impute=columns_to_impute,
                            special_impute=special_column,
                            representation=None,
                            structural_features=None,
                            unroll=None,
                            numerical_feats=numerical_feats,
                            target_features=target_features,
                            regressor_type=regressor_type,
                            transform_type=transform_type,
                            hyperparameter_optimization=hyperparameter_optimization,
                            imputer=imputer
                        )
    save_results(scores,
                predictions=predictions,
                imputer=imputer,
                representation= None,
                pu_type= None,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                TEST=TEST
                )



# save_results(scores: dict,
#                  predictions: pd.DataFrame,
#                  target_features: str,
#                  regressor_type: str,
#                  TEST : bool =True,
#                  representation: str=None,
#                  pu_type : Optional[str]=None,
#                  radius : Optional[int]=None,
#                  vector : Optional[str]=None,
#                  numerical_feats: Optional[list[str]]=None,
#                  imputer: Optional[str] = None,
#                  output_dir_name: str = "results",
#                  ) -> None:

# perform model

def perform_model_numerical(regressor_type:str):
        
        main_numerical_only(dataset=w_data,
                            regressor_type=regressor_type,
                            transform_type= "Standard",
                            hyperparameter_optimization= True,
                            target_features= ['Lp (nm)'],
                            )


# perform numerical and mordred

# def main_mordred_numerical(
#     dataset: pd.DataFrame,
#     regressor_type: str,
#     target_features: list[str],
#     transform_type: str,
#     hyperparameter_optimization: bool,
#     oligomer_representation: str

# ) -> None:

#     columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
#     special_column: str = "Mw (g/mol)"
#     numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]

#     imputer = "mean"
#     scores, predictions  = train_regressor(
#                             dataset=dataset,
#                             features_impute=columns_to_impute,
#                             special_impute=special_column,
#                             representation=None,
#                             structural_features=None,
#                             unroll=None,
#                             numerical_feats=numerical_feats,
#                             target_features=target_features,
#                             regressor_type=regressor_type,
#                             transform_type=transform_type,
#                             hyperparameter_optimization=hyperparameter_optimization,
#                             imputer=imputer
#                         )
#     save_results(scores,
#                 predictions=predictions,
#                 representation= None,
#                 pu_type= None,
#                 target_features=target_features,
#                 regressor_type=regressor_type,
#                 numerical_feats=numerical_feats,
#                 TEST=TEST
#                 )




# perfrom numerical and maccs

def main_maccs_numerical(
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

    columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    special_column: str = "Mw (g/mol)"
    numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    imputer = "mean"

    scores, predictions  = train_regressor(
                            dataset=dataset,
                            features_impute=columns_to_impute,
                            special_impute=special_column,
                            representation=representation,
                            structural_features=structural_features,
                            unroll=unroll_single_feat,
                            numerical_feats=numerical_feats,
                            target_features=target_features,
                            regressor_type=regressor_type,
                            transform_type=transform_type,
                            hyperparameter_optimization=hyperparameter_optimization,
                            imputer=imputer
                        )
    save_results(scores,
                predictions=predictions,
                imputer=imputer,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                TEST=TEST
                )


def perform_model_numerical_maccs(regressor_type:str):
        for oligo_type in edited_oligomer_list: 
            main_maccs_numerical(dataset=w_data,
                                regressor_type=regressor_type,
                                transform_type= "Standard",
                                hyperparameter_optimization= True,
                                target_features= ['Lp (nm)'],
                                oligomer_representation=oligo_type
                                )






def main_ecfp_numerical(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
) -> None:
    # representation: str = "MACCS"
    # structural_features: list[str] = [f"{oligomer_representation}_{representation}"]
    # unroll_single_feat = {"representation": representation,
    #                       "oligomer_representation":oligomer_representation,
    #                       "col_names": structural_features}

    columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    special_column: str = "Mw (g/mol)"
    numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]

    imputer = "mean"
    scores, predictions  = train_regressor(
                            dataset=dataset,
                            features_impute=columns_to_impute,
                            special_impute=special_column,
                            representation=None,
                            structural_features=None,
                            unroll=None,
                            numerical_feats=numerical_feats,
                            target_features=target_features,
                            regressor_type=regressor_type,
                            transform_type=transform_type,
                            hyperparameter_optimization=hyperparameter_optimization,
                            imputer=imputer
                        )
    save_results(scores,
                predictions=predictions,
                representation= None,
                pu_type= None,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                TEST=TEST
                )








if __name__ == "__main__":
    parser = ArgumentParser(description="Run models on HPC")
    parser.add_argument('--model', type=str, choices=['RF', 'MLR'], required=True, help="Specify the model to run")
    parser.add_argument('--function', type=str, choices=['numerical', 'numerical_maccs'], required=True, help="Specify the function to run")

    args = parser.parse_args()
    model = args.model
    function = args.function

    if function == 'numerical':
        perform_model_numerical(model)
    elif function == 'numerical_maccs':
        perform_model_numerical_maccs(model)



