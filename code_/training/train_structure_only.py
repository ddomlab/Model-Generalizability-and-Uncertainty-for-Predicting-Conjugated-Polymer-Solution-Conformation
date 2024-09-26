import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits,cutoffs
import sys
import json
import numpy as np
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

TEST=False

def main_ECFP_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    radius: int,
    oligomer_representation: str,
    vector_type: str
) -> None:
    representation: str = "ECFP"
    n_bits = radius_to_bits[radius]
    structural_features: list[str] = [
        f"{oligomer_representation}_{representation}{2 * radius}_{vector_type}_{n_bits}bits"
    ]
    unroll_single_feat = {
        "representation": representation,
        "radius": radius,
        "n_bits": n_bits,
        "vector_type": vector_type,
        "oligomer_representation":oligomer_representation,
        "col_names": structural_features,
    }

    scores, predictions = train_regressor(
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
                            cutoff=cutoffs,
                            imputer=None
                        )
    save_results(scores,
            predictions=predictions,
            representation= representation,
            pu_type= oligomer_representation,
            radius= radius,
            vector =vector_type,
            target_features=target_features,
            regressor_type=regressor_type,
            cutoff=cutoffs,
            TEST=TEST
            )


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
                            cutoff=cutoffs,
                            imputer=None
                        )


    save_results(scores,
                predictions=predictions,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                cutoff=cutoffs,
                TEST=TEST
                )


def main_Mordred_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    oligomer_representation: str
) -> None:
    representation: str = "Mordred"
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
                            cutoff=cutoffs,
                            imputer=None
                        )

    save_results(scores,
                predictions=predictions,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                cutoff=cutoffs,
                TEST=TEST
                )



training_df_dir: Path = DATASETS/ "training_dataset"/ "structure_wo_block_cp_scaler_dataset.pkl"
oligo_dir: Path = DATASETS/ "raw"/"pu_columns_used.json"

oligomer_list =open_json(oligo_dir)
w_data = pd.read_pickle(training_df_dir)
edited_oligomer_list = [" ".join(x.split()[:-1]) for x in oligomer_list]


# for radius in radii:
# for vector in vectors:
# radii = [3, 4, 5, 6]
# vectors = ['count', 'binary']
def perform_model_ecfp(radius,vector):
    for oligo_type in edited_oligomer_list:
                print(f'polymer unit :{oligo_type} with rep of ECFP{radius} and {vector}')
                main_ECFP_only(
                                dataset=w_data,
                                regressor_type= 'RF',
                                target_features= ['Lp (nm)'],
                                transform_type= "Standard",
                                hyperparameter_optimization= True,
                                radius = radius,
                                oligomer_representation=oligo_type,
                                vector_type=vector
                                )




def perform_model_maccs():
    for oligo_type in edited_oligomer_list:
            main_MACCS_only(
                            dataset=w_data,
                            regressor_type= 'RF',
                            target_features= ['Lp (nm)'],
                            transform_type= "Standard",
                            hyperparameter_optimization= True,
                            oligomer_representation=oligo_type,
                            )
 


# Rg1 (nm)
def perform_model_mordred():
    for oligo_type in edited_oligomer_list:
                main_Mordred_only(
                                dataset=w_data,
                                regressor_type= 'RF',
                                target_features= ['Lp (nm)'],
                                transform_type= "Standard",
                                hyperparameter_optimization= True,
                                oligomer_representation=oligo_type,
                                )




# perform_model_ecfp()
# perform_model_maccs()
# perform_model_mordred()

def main():
    parser = ArgumentParser(description="Run different models")
    
    parser.add_argument(
        '--model',
        choices=['mordred', 'maccs', 'ecfp'],
        required=True,
        help='Specify which model to run: mordred, maccs, or ecfp'
    )
    
    parser.add_argument(
        '--radius',
        type=int,
        help='Radius for ECFP (only used for ecfp model)'
    )
    
    parser.add_argument(
        '--vector',
        choices=['count', 'binary'],
        help='Vector type for ECFP (only used for ecfp model)'
    )
    
    args = parser.parse_args()

    if args.model == 'mordred':
        perform_model_mordred()
    elif args.model == 'maccs':
        perform_model_maccs()
    elif args.model == 'ecfp':
        if args.radius is None or args.vector is None:
            parser.error('--radius and --vector are required for the ecfp model')
        perform_model_ecfp(args.radius, args.vector)

if __name__ == '__main__':
    main()

