import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits,cutoffs
from typing import Callable, Optional, Union, Dict, Tuple
import numpy as np
import sys
sys.path.append("../cleaning")
from argparse import ArgumentParser
from data_handling import save_results

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training_dataset"/ "Rg data with clusters.pkl"
w_data = pd.read_pickle(training_df_dir)

TEST = False

def main_numerical_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    numerical_feats: Optional[list[str]],
    columns_to_impute: Optional[list[str]]=None,
    special_impute: Optional[str]=None,
    imputer:Optional[str]=None,
    kernel:str=None,
    cutoff:Optional[str]=None,
    second_transformer:str=None,
    classification:bool=False,

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
                                            kernel=kernel,
                                            transform_type=transform_type,
                                            cutoff=cutoff,
                                            hyperparameter_optimization=hyperparameter_optimization,
                                            imputer=imputer,
                                            second_transformer=second_transformer,
                                            Test=TEST,
                                            classification=classification,
                                            )
    
    save_results(scores,
                predictions=predictions,
                df_shapes=data_shapes,
                imputer=imputer,
                representation= None,
                pu_type= None,
                target_features=target_features,
                regressor_type=regressor_type,
                kernel=kernel,
                numerical_feats=numerical_feats,
                cutoff=cutoffs,
                TEST=TEST,
                hypop=hyperparameter_optimization,
                transform_type=transform_type,
                second_transformer=second_transformer,
                classification=classification,
                special_name='hp_RF_differences'
                )


    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # imputer = "mean"
    # transform_type= "Standard"
    # target_features= ['Lp (nm)']
    



# perform_model_numerical_maccs('RF')

def parse_arguments():
    parser = ArgumentParser(description="Process some data for numerical-only regression.")
    
    # Argument for regressor_type
    parser.add_argument(
        '--target_features',
        # choices=['Lp (nm)', 'Rg1 (nm)', 'Rh (IW avg log)'],  
        required=True,
        help="Specify a single target for the analysis."
    )
    
    parser.add_argument(
        '--regressor_type', 
        type=str, 
        # choices=['RF', 'DT', 'MLR', 'SVR', 'XGBR','KNN', 'GPR', 'NGB', 'sklearn-GPR'], 
        required=True, 
        help="Regressor type required"
    )

    parser.add_argument(
        '--numerical_feats',
        type=str,
        choices=['Xn','Mn (g/mol)', 'Mw (g/mol)', 'PDI', 'Temperature SANS/SLS/DLS/SEC (K)',
                  'Concentration (mg/ml)','solvent dP',	'polymer dP',	'solvent dD',	'polymer dD',	'solvent dH',	'polymer dH', 'Ra',
                  "abs(solvent dD - polymer dD)", "abs(solvent dP - polymer dP)", "abs(solvent dH - polymer dH)"],

        nargs='+',  # Allows multiple choices
        required=True,
        help="Numerical features: choose"
    )
    
    parser.add_argument(
        '--columns_to_impute',
        type=str,
        choices=['Mn (g/mol)', 'Mw (g/mol)', 'PDI', 'Temperature SANS/SLS/DLS/SEC (K)',
                  'Concentration (mg/ml)','solvent dP',	'polymer dP',	'solvent dD',	'polymer dD',	'solvent dH',	'polymer dH', 'Ra'],

        nargs='*',  # This allows 0 or more values
        default=None,  
        help="imputation features: choose"
    )

    parser.add_argument(
        '--imputer',
        choices=['mean', 'median', 'most_frequent',"distance KNN", None],  
        nargs='?',  # This allows the argument to be optional
        default=None,  
        help="Specify the imputation strategy or leave it as None."
    )

    parser.add_argument(
        '--special_impute',
        choices=['Mw (g/mol)', None],  
        nargs='?',  # This allows the argument to be optional
        default=None,  # Set the default value to None
        help="Specify the imputation strategy or leave it as None."
    )

    parser.add_argument(
        "--transform_type", 
        type=str, 
        choices=["Standard", "Robust Scaler"], 
        default= "Standard", 
        help="transform type required"
    )

    parser.add_argument(
        "--kernel", 
        type=str,
        default=None,
        help='kernel for GP is optinal'
    )


    return parser.parse_args()

if __name__ == "__main__":
    # args = parse_arguments()
    

    # main_numerical_only(
    #     dataset=w_data,
    #     regressor_type=args.regressor_type,
    #     kernel=args.kernel,
    #     target_features=[args.target_features],  
    #     transform_type='Standard',
    #     hyperparameter_optimization=True,
    #     columns_to_impute=args.columns_to_impute,  
    #     special_impute=args.special_impute,
    #     numerical_feats=args.numerical_feats,  
    #     imputer=args.imputer,
    #     cutoff=None,  
    #     # second_transformer='Log',
    #     classification=False
    # )

    main_numerical_only(
        dataset=w_data,
        regressor_type="RF",
        # kernel= "matern",
        target_features=['log Rg (nm)'],  # Can adjust based on actual usage
        transform_type='Standard',
        hyperparameter_optimization=True,
        columns_to_impute=None,
        special_impute=None,
        numerical_feats=['Mw (g/mol)','PDI', "Concentration (mg/ml)", "Temperature SANS/SLS/DLS/SEC (K)", "solvent dP", "solvent dD", "solvent dH"],
        imputer=None,
        classification=False,
        cutoff=None)

    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]

# "intensity weighted average over log(Rh (nm))"



