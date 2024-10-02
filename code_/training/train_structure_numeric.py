import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits,cutoffs
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

training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.pkl"
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
    columns_to_impute: list[str],
    special_impute: str,
    numerical_feats: list[str],
    imputer:str,
    cutoff:str=None
) -> None:


    scores, predictions,data_shapes  = train_regressor(
                                            dataset=dataset,
                                            features_impute=columns_to_impute,
                                            special_impute=special_impute,
                                            representation=None,
                                            structural_features=None,
                                            unroll=None,
                                            numerical_feats=numerical_feats,
                                            target_features=target_features,
                                            regressor_type=regressor_type,
                                            transform_type=transform_type,
                                            cutoff=cutoff,
                                            hyperparameter_optimization=hyperparameter_optimization,
                                            imputer=imputer
                                            )
    
    save_results(scores,
                predictions=predictions,
                df_shapes=data_shapes,
                imputer=imputer,
                representation= None,
                pu_type= None,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                cutoff=cutoffs,
                TEST=TEST
                )


    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # imputer = "mean"
    transform_type= "Standard"
    target_features= ['Lp (nm)']
    
# def perform_model_numerical(regressor_type:str):
        
# main_numerical_only(dataset=w_data,
#                     regressor_type=regressor_type,
#                     transform_type= "Standard",
#                     hyperparameter_optimization= True,
#                     target_features= ['Lp (nm)'],
#                     columns_to_impute,
#                     special_column,
#                     numerical_feats,
#                     imputer,
#                     )


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
                            cutoff=cutoffs,
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
                cutoff=cutoffs,
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
    scores, predictions, data_shapes  = train_regressor(
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
                                                cutoff=cutoffs,
                                                hyperparameter_optimization=hyperparameter_optimization,
                                                imputer=imputer
                                            )
    save_results(scores,
                predictions=predictions,
                df_shapes = data_shapes,
                representation= None,
                pu_type= None,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                cutoff=cutoffs,
                TEST=TEST
                )




# perform_model_numerical_maccs('RF')

def parse_arguments():
    parser = ArgumentParser(description="Process some data for numerical-only regression.")
    
    # Argument for regressor_type
    parser.add_argument(
        '--target_features',
        choices=['Lp (nm)', 'Rg1 (nm)', 'Rh1 (nm)'],  # Valid choices for target
        required=True,
        help="Specify a single target for the analysis."
    )
    
    parser.add_argument(
        '--regressor_type', 
        type=str, 
        choices=['RF', 'DT', 'MLR', 'SVR', 'XGBR','KNN'], 
        required=True, 
        help="Regressor type: RF, DT, or MLR."
    )


    parser.add_argument(
        '--numerical_feats',
        type=str,
        nargs='+',  # Allows multiple choices
        required=True,
        help="Numerical features: choose as a space-separated list"
    )
    

    parser.add_argument(
        '--columns_to_impute',
        type=str,
        nargs='*',  # This allows 0 or more values
        default=None,  
        help="Imputation features: choose as a space-separated list"
    )

    parser.add_argument(
        '--imputer',
        choices=['mean', 'median', 'most_frequent',"distance KNN", None],  # Add 'None' as an option
        nargs='?',  # This allows the argument to be optional
        default='mean',  # Set the default value to None
        help="Specify the imputation strategy or leave it as None."
    )

    parser.add_argument(
        '--special_impute',
        choices=['Mw (g/mol)', None],  # Add 'None' as an option
        nargs='?',  # This allows the argument to be optional
        default=None,  # Set the default value to None
        help="Specify the imputation strategy or leave it as None."
    )
    args = parser.parse_args()

    # Process numerical_feats and columns_to_impute (splitting by commas)
    args.numerical_feats = [feat.strip() for feat in args.numerical_feats.split(",")]
    args.columns_to_impute = [col.strip() for col in args.columns_to_impute.split(",")] if args.columns_to_impute else []
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Call the main function with parsed arguments
    main_numerical_only(
        dataset=w_data,
        regressor_type=args.regressor_type,
        target_features=[args.target_features],  # Already a list from `choices`, no need to wrap
        transform_type="Standard",
        hyperparameter_optimization=True,
        columns_to_impute=args.columns_to_impute,  # Already a list
        special_impute=args.special_impute,
        numerical_feats=args.numerical_feats,  # Already a list
        imputer=args.imputer,
        cutoff=None,  # Optional cutoff value
    )


# main_numerical_only(
#     dataset=w_data,
#     regressor_type="DT",
#     target_features=["Rg1 (nm)"],  # Can adjust based on actual usage
#     transform_type="Standard",
#     hyperparameter_optimization=True,
#     columns_to_impute=["Concentration (mg/ml)"],
#     special_impute=None,
#     numerical_feats=["Concentration (mg/ml)",'Temperature SANS/SLS/DLS/SEC (K)'],
#     imputer="mean",
#     cutoff=cutoffs)


# if __name__ == "__main__":
#     parser = ArgumentParser(description="Run models on HPC")
#     parser.add_argument('--model', type=str, choices=['RF', 'MLR'], required=True, help="Specify the model to run")
#     parser.add_argument('--function', type=str, choices=['numerical', 'numerical_maccs'], required=True, help="Specify the function to run")

#     args = parser.parse_args()
#     model = args.model
#     function = args.function

#     if function == 'numerical':
#         perform_model_numerical(model)
#     elif function == 'numerical_maccs':
#         perform_model_numerical_maccs(model)



