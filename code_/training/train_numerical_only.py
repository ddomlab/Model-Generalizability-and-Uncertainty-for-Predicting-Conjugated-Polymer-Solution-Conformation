import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits, cutoffs
from typing import Callable, Optional, Union, Dict, Tuple
import numpy as np
import sys
sys.path.append("../cleaning")
from argparse import ArgumentParser
from data_handling import save_results
from train_structure_numerical import parse_arguments

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training_dataset"/ "non_imputed_full_Rg_data.pkl"
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


    scores, predictions  = train_regressor(
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
    save_results(
                scores=scores,
                predictions=predictions,
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
                # special_folder_name='hp_RF_differences'
                special_file_name='Revision',
                )


    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # imputer = "mean"
    # transform_type= "Standard"
    # target_features= ['Lp (nm)']
    


if __name__ == "__main__":
    # if TEST==False:

    #     args = parse_arguments()
    #     main_numerical_only(
    #         dataset=w_data,
    #         regressor_type=args.regressor_type,
    #         kernel=args.kernel,
    #         target_features=[args.target_features],  
    #         transform_type=args.transform_type,
    #         hyperparameter_optimization=True,
    #         columns_to_impute=args.columns_to_impute,  
    #         special_impute=args.special_impute,
    #         numerical_feats=args.numerical_feats,  
    #         imputer=args.imputer,
    #         cutoff=None,  
    #         second_transformer=None,
    #         classification=False
    #     )
    # else:
        # print(w_data['SANS/SAXS model'].isnull().sum())
        # w_data = w_data[w_data["Concentration (mg/ml)"] < 30]
        # w_data["log concentration"] = np.log10(w_data["Concentration (mg/ml)"])
        # w_data["log Xn"] = np.log10(w_data["Xn"])
        # w_data["log Mw"] = np.log10(w_data["Mw (g/mol)"])
        # w_data["log PDI"] = np.log10(w_data["PDI"])
        # w_data["log Temperature"] = np.log10(w_data["Temperature SANS/SLS/DLS/SEC (K)"])
        
        # w_data["model_fitting_encoded"]=w_data['SANS/SAXS model'].astype("category").cat.codes
        # print(w_data["model_fitting_encoded"].isnull().sum())
        def encode_light_dark(value):
            value_str = str(value).strip().lower()
            if value_str == 'light':
                return 1
            elif value_str == 'dark' or value_str =='dark2':
                return 0
            else:
                return value
            
        ##### Some cleaning for converting str to numerical like overnight -> 12 h, 4-5 -> mean of 4,5.
        def convert_str_to_num(value, colname):
            if isinstance(value, (int, float)):
                return value
            if '-' in str(value) or '−' in str(value):  # Added handling for en dash character
                parts = str(value).replace('−', '-').split('-')  # Replacing en dash with hyphen
                if len(parts) == 2:  # If it's in the format number1-number2
                    num1, num2 = map(float, parts)
                    return (num1 + num2) / 2
            if 'overnight' in str(value):
                if 'min' in colname.lower():
                    return 16 * 60  # 12 hours in minutes
                elif 'hour' in colname.lower():
                    return 16
            else:
                return value  
            
        env_processing_parameters = [ "Dark/light", "Aging time (hour)", "To Aging Temperature (K)",
                            "Sonication/Stirring/heating Temperature (K)", "Merged Stirring /sonication/heating time(min)","Storage time (hour)"]
        w_data['Dark/light'] = w_data['Dark/light'].apply(encode_light_dark)
        for col in env_processing_parameters:
                w_data[col] = w_data[col].apply(lambda val: convert_str_to_num(val, col))
                print(f"{col}: {w_data[col].dtype}")
        w_data["Aging time (hour)"].fillna(w_data["Storage time (hour)"], inplace=True)
        for m in ["RF", "XGBR"]:
            main_numerical_only(
                dataset=w_data,
                regressor_type=m,
                # kernel= "matern",
                target_features=['log Rg (nm)'],  # Can adjust based on actual usage
                transform_type='Standard',  
                hyperparameter_optimization=False,
                columns_to_impute=["PDI", "Temperature SANS/SLS/DLS/SEC (K)", "Concentration (mg/ml)",
                                "Dark/light", "Aging time (hour)", "To Aging Temperature (K)",
                                "Sonication/Stirring/heating Temperature (K)", "Merged Stirring /sonication/heating time(min)"
                                ],
                special_impute='Mw (g/mol)',
                numerical_feats=['Xn', 'Mn (g/mol)', 'Mw (g/mol)', 'PDI', 'Concentration (mg/ml)', 'Temperature SANS/SLS/DLS/SEC (K)',
                                "polymer dP", "polymer dD", "polymer dH", 'solvent dP', 'solvent dD', 'solvent dH',
                                "Dark/light", "Aging time (hour)", "To Aging Temperature (K)",
                                "Sonication/Stirring/heating Temperature (K)", "Merged Stirring /sonication/heating time(min)",
                                #   "model_fitting_encoded"
                                ],
                imputer="distance KNN_7",
                classification=False,
                cutoff=None)


            # pfo_pht_data = w_data[w_data['canonical_name'].isin(['rr-P3HT', 'PFO'])]
            # main_numerical_only(
            # dataset=pfo_pht_data,
            # regressor_type="MLR",
            # # kernel= "matern",
            # target_features=['log Rg (nm)'],  # Can adjust based on actual usage
            # transform_type=None,  
            # hyperparameter_optimization=False,
            # columns_to_impute=None,
            # special_impute=None,
            # numerical_feats=['Xn'],
            # imputer=None,
            # classification=False,
            # cutoff=None)

    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]

# "intensity weighted average over log(Rh (nm))"



