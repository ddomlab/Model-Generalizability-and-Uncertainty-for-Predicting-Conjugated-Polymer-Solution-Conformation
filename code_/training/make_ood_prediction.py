import pandas as pd
from pathlib import Path
from get_ood_split import train_regressor
from train_structure_numerical import get_structural_info
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

training_df_dir: Path = DATASETS/ "training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal (40-1000 nm)_added.pkl"
w_data = pd.read_pickle(training_df_dir)




TEST = False

# def get_structural_info(fp:str,poly_unit:str,radius:int=None,vector:str=None)->Tuple:
       
#         if fp == "Mordred" or fp == "MACCS":
#             fp_features: list[str] = [f"{poly_unit}_{fp}"]
#             unrolling_featurs = {"representation": fp,
#                                 "oligomer_representation":poly_unit,
#                                 "col_names": fp_features}
#             return fp_features, unrolling_featurs
        
#         if fp == "ECFP":
#             n_bits = radius_to_bits[radius]
#             fp_features: list[str] = [
#             f"{poly_unit}_{fp}{2 * radius}_{vector}_{n_bits}bits"]
#             unrolling_featurs = {
#                                 "representation": fp,
#                                 "radius": radius,
#                                 "n_bits": n_bits,
#                                 "vector_type": vector,
#                                 "oligomer_representation": poly_unit,
#                                 "col_names": fp_features,
#                             }
#             return fp_features, unrolling_featurs
#         else:
#               return None, None


def main_structural_numerical(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    numerical_feats: Optional[list[str]],
    representation:str,
    oligomer_representation: str,
    hyperparameter_optimization: bool=True,
    radius:int=None,
    vector:str=None,
    kernel:str = None,
    cutoff:Optional[str]=None,
    second_transformer:str=None,
    clustering_method:str=None,
) -> None:
    
    structural_features, unroll_single_feat = get_structural_info(representation,oligomer_representation,radius,vector)
    scores, predictions,cluster_y_truth  = train_regressor(
                                                    dataset=dataset,
                                                    representation=representation,
                                                    structural_features=structural_features,
                                                    unroll=unroll_single_feat,
                                                    numerical_feats=numerical_feats,
                                                    target_features=target_features,
                                                    regressor_type=regressor_type,
                                                    kernel=kernel,
                                                    transform_type=transform_type,
                                                    second_transformer=second_transformer,
                                                    cutoff=cutoff,
                                                    hyperparameter_optimization=hyperparameter_optimization,
                                                    clustering_method=clustering_method,
                                                    Test=TEST,
                                                )
    save_results(scores,
                predictions=predictions,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                radius= radius,
                vector =vector,
                kernel=kernel,
                cutoff=cutoff,
                TEST=TEST,
                hypop=hyperparameter_optimization,
                transform_type=transform_type,
                second_transformer=second_transformer,
                clustering_method=clustering_method
                )
    




if __name__ == "__main__":
    # args = parse_arguments()

    # main_structural_numerical(
    #     dataset=w_data,
    #     representation=args.representation,
    #     radius=args.radius,
    #     vector=args.vector,
    #     oligomer_representation = args.oligomer_representation,
    #     regressor_type=args.regressor_type,
    #     kernel=args.kernel,
    #     target_features=[args.target_features],  
    #     transform_type=None,
    #     second_transformer='Log',
    #     hyperparameter_optimization=True,
    #     columns_to_impute=args.columns_to_impute,  
    #     special_impute=args.special_impute,
    #     numerical_feats=args.numerical_feats,  
    #     imputer=args.imputer,
    #     cutoff=None,  
    # )


        main_structural_numerical(
        dataset=w_data,
        representation="MACCS",
        # radius=3,
        # vector="count",
        regressor_type="RF",
        target_features=['Rg1 (nm)'],  
        transform_type=None,
        second_transformer=None,
        numerical_feats=['Mw (g/mol)', 'Concentration (mg/ml)', 'Temperature SANS/SLS/DLS/SEC (K)', 'solvent dP', 'solvent dD', 'solvent dH'],
        hyperparameter_optimization=True,
        oligomer_representation="Monomer",
        clustering_method="KMeans"
    )