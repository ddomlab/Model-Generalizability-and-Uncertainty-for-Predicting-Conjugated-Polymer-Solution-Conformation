import pandas as pd
from pathlib import Path
from get_ood_split_learning_curve import train_ood_learning_curve
from train_structure_numerical import get_structural_info
from typing import Callable, Optional, Union, Dict, Tuple
import numpy as np
import sys
import os

from argparse import ArgumentParser
from data_handling import save_results
from train_structure_numerical import parse_arguments
HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training_dataset"/"Rg data with clusters.pkl"
w_data = pd.read_pickle(training_df_dir)

# clusters = 'KM4 ECFP6_Count_512bit cluster'	
# 'KM3 Mordred cluster'	
# 'substructure cluster'
# 'EG-Ionic-Based Cluster'
# 'KM5 polymer_solvent HSP and polysize cluster'	
# 'KM4 polymer_solvent HSP cluster'
# 'KM4 Mordred_Polysize cluster'


TEST = True


def main_structural_numerical(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    numerical_feats: Optional[list[str]],
    representation:str=None,
    oligomer_representation: str=None,
    radius:int=None,
    vector:str=None,
    second_transformer:str=None,
    clustering_method:str=None,
) -> None:
    
    structural_features, unroll_single_feat = get_structural_info(representation,oligomer_representation,radius,vector)
    scores, predictions  = train_ood_learning_curve(
                                                    dataset=dataset,
                                                    target_features=target_features,
                                                    representation=representation,
                                                    structural_features=structural_features,
                                                    unroll=unroll_single_feat,
                                                    numerical_feats=numerical_feats,
                                                    regressor_type=regressor_type,
                                                    transform_type=transform_type,
                                                    second_transformer=second_transformer,
                                                    clustering_method=clustering_method,
                                                    Test=TEST,
                                                )

    _ = save_results(
                    scores,
                    predictions=predictions,
                    representation= representation,
                    pu_type= oligomer_representation,
                    target_features=target_features,
                    regressor_type=regressor_type,
                    numerical_feats=numerical_feats,
                    radius= radius,
                    vector =vector,
                    TEST=TEST,
                    hypop=False,
                    transform_type=transform_type,
                    second_transformer=second_transformer,
                    clustering_method=clustering_method,
                    learning_curve=True,
                    )
    
    # scores_criteria: list= ['mae', 'rmse',
    #                         'r2', 'spearman_r']

    # suffix = f"{regressor_type}_{transform_type}" if transform_type else f'{regressor_type}'                        
    # suffix = f"{suffix}_{representation}" if representation else suffix
    # feats_abbv = generate_acronym_string(numerical_feats) if numerical_feats else None
    # suffix = f"{suffix}_{feats_abbv}" if feats_abbv else suffix
    # score_plot_folder = saving_folder/ f'comparitive cluster scores ({suffix})'
    # plot_splits_scores(scores=scores, scores_criteria=scores_criteria, folder=score_plot_folder)

    # print("-"*30
    #       ,"\nPlotted Comparitive Cluster Scores!")
    

    # parity_folder = saving_folder/ f'parity plot ({suffix})'

    
    # print("_"*30,
    #       "\nPlotted Parity Plots!")


if __name__ == "__main__":
    args = parse_arguments()

    main_structural_numerical(
        dataset=w_data,
        representation=args.representation,
        radius=args.radius,
        vector=args.vector,
        oligomer_representation = args.oligomer_representation,
        regressor_type=args.regressor_type,
        target_features=[args.target_features],  
        transform_type='Standard',
        second_transformer=None,
        numerical_feats=args.numerical_feats, 
        clustering_method=args.clustering_method,
    )


# # TODO: Update the dataset to include the clustering methods

        # main_structural_numerical(
        # dataset=w_data,
        # representation="MACCS",
        # # radius=3,
        # # vector="count",
        # regressor_type="NGB",
        # target_features=['log Rg (nm)'],  
        # transform_type='Standard',
        # second_transformer=None,
        # numerical_feats=['Mw (g/mol)', 'PDI', 'Concentration (mg/ml)', 'Temperature SANS/SLS/DLS/SEC (K)', 'polymer dP', 'polymer dD' , 'polymer dH', 'solvent dP', 'solvent dD', 'solvent dH'],
        # oligomer_representation="Trimer",
        # clustering_method='substructure cluster'
        # )
        
        # main_structural_numerical(
        # dataset=w_data,
        # representation=None,
        # # radius=3,
        # # vector="count",
        # regressor_type="RF",
        # target_features=['log Rg (nm)'],  
        # transform_type='Standard',
        # second_transformer=None,
        # numerical_feats=['Mw (g/mol)', 'PDI', 'Concentration (mg/ml)', 'Temperature SANS/SLS/DLS/SEC (K)', 'polymer dP', 'polymer dD' , 'polymer dH', 'solvent dP', 'solvent dD', 'solvent dH'],
        # oligomer_representation=None,
        # clustering_method='substructure cluster'
        # )