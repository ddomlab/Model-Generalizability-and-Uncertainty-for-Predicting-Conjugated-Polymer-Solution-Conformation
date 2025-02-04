import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits,cutoffs
import sys
# import numpy as np
from typing import Callable, Optional, Union, Dict, Tuple
sys.path.append("../cleaning")
# from clean_dataset import open_json
from argparse import ArgumentParser
from data_handling import save_results
from train_structure_numerical import get_structural_info

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal (40-1000 nm)_added.pkl"
w_data = pd.read_pickle(training_df_dir)


TEST=True



def main_structural(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    representation:str,
    oligomer_representation: str,
    hyperparameter_optimization: bool=True,
    radius:int=None,
    vector:str=None,
    kernel:str = None,
    cutoff:Optional[str]=None,
    second_transformer:str=None,
) -> None:
    
    structural_features, unroll_single_feat = get_structural_info(representation,oligomer_representation,radius,vector)
    scores, predictions,data_shapes  = train_regressor(
                                                    dataset=dataset,
                                                    representation=representation,
                                                    structural_features=structural_features,
                                                    unroll=unroll_single_feat,
                                                    features_impute=None,
                                                    special_impute=None,
                                                    numerical_feats=None,
                                                    imputer=None,
                                                    target_features=target_features,
                                                    regressor_type=regressor_type,
                                                    kernel=kernel,
                                                    transform_type=transform_type,
                                                    second_transformer=second_transformer,
                                                    cutoff=cutoff,
                                                    hyperparameter_optimization=hyperparameter_optimization,
                                                    Test=TEST,
                                                )
    save_results(scores,
                predictions=predictions,
                df_shapes=data_shapes,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                radius= radius,
                vector =vector,
                kernel=kernel,
                cutoff=cutoff,
                TEST=TEST,
                hypop=hyperparameter_optimization,
                transform_type=transform_type,
                second_transformer=second_transformer,
                )







def main_ECFP_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    radius: int,
    oligomer_representation: str,
    vector_type: str,
    kernel:Optional[str]=None,
    
    
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
    scores, predictions, data_shapes = train_regressor(
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
                                        imputer=None,
                                        kernel=kernel,
                                        Test=TEST
                                    )
    save_results(scores,
            predictions=predictions,
            df_shapes=data_shapes,
            generalizability_score=None,
            representation= representation,
            pu_type= oligomer_representation,
            radius= radius,
            vector =vector_type,
            target_features=target_features,
            regressor_type=regressor_type,
            kernel=kernel,
            cutoff=cutoffs,
            TEST=TEST,
            transform_type=transform_type
            )



def main_MACCS_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    oligomer_representation: str,
    kernel:Optional[str]=None,
) -> None:
    representation: str = "MACCS"
    structural_features: list[str] = [f"{oligomer_representation}_{representation}"]
    unroll_single_feat = {"representation": representation,
                          "oligomer_representation":oligomer_representation,
                          "col_names": structural_features}

    scores, predictions, data_shapes  = train_regressor(
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
                                            imputer=None,
                                            kernel=kernel,
                                            Test=TEST
                                        )


    save_results(scores,
                predictions=predictions,
                df_shapes=data_shapes,
                generalizability_score=None,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                cutoff=cutoffs,
                TEST=TEST,
                transform_type=transform_type
                )


def main_Mordred_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    oligomer_representation: str,
    kernel:Optional[str]=None,
) -> None:
    representation: str = "Mordred"
    structural_features: list[str] = [f"{oligomer_representation}_{representation}"]
    unroll_single_feat = {"representation": representation,
                          "oligomer_representation":oligomer_representation,
                          "col_names": structural_features}

    scores, predictions, data_shapes  = train_regressor(
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
                                    imputer=None,
                                    kernel=kernel,
                                    Test=TEST
                                )

    save_results(scores=scores,
                predictions=predictions,
                df_shapes=data_shapes,
                generalizability_score=None,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                kernel=kernel,
                cutoff=cutoffs,
                TEST=TEST,
                transform_type=transform_type,
                # hyperparameter_optimization=hyperparameter_optimization,
                )






# for radius in radii:
# for vector in vectors:
# radii = [3, 4, 5, 6]
# vectors = ['count', 'binary']
def perform_model_ecfp(regressor_type:str,
                        radius:int,vector:str,
                        target:str,
                        oligo_type:str,
                        kernel:Optional[str]=None,
                        transform_type:Optional[str]='Standard'):
    # for oligo_type in edited_oligomer_list:
                print(oligo_type)
                main_ECFP_only(
                                dataset=w_data,
                                regressor_type= regressor_type,
                                target_features= [target],
                                transform_type= transform_type,
                                hyperparameter_optimization= True,
                                radius = radius,
                                oligomer_representation=oligo_type,
                                vector_type=vector,
                                kernel=kernel
                                )




def perform_model_maccs(regressor_type:str,target:str,oligo_type:str, kernel:Optional[str]=None, transform_type:Optional[str]='Standard'):
    # for oligo_type in edited_oligomer_list:
            print(oligo_type)
            main_MACCS_only(
                            dataset=w_data,
                            regressor_type= regressor_type,
                            target_features= [target],
                            transform_type= transform_type,
                            hyperparameter_optimization= True,
                            oligomer_representation=oligo_type,
                            kernel=kernel,
                            )
 


# Rg1 (nm)
def perform_model_mordred(regressor_type:str,target:str,oligo_type:str,kernel:Optional[str]=None, transform_type:Optional[str]='Standard'):
                print(oligo_type)
                main_Mordred_only(
                                dataset=w_data,
                                regressor_type= regressor_type,
                                target_features= [target],
                                transform_type= transform_type,
                                hyperparameter_optimization= False,
                                oligomer_representation=oligo_type,
                                kernel=kernel
                                )




# perform_model_ecfp('GPR',6,"count",'Rg1 (nm)', 'Monomer',kernel='tanimoto')
# perform_model_maccs()
# perform_model_mordred('NGB',"multimodal Rh", 'Monomer', transform_type="Robust Scaler")

# def main():
#     parser = ArgumentParser(description='Run models with specific parameters')

#     # Subparsers for different models
#     subparsers = parser.add_subparsers(dest='model', required=True, help='Choose a model to run')

#     # Parser for ECFP model
#     parser_ecfp = subparsers.add_parser('ecfp', help='Run the ECFP model')
#     parser_ecfp.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
#     parser_ecfp.add_argument('--radius', type=int, choices=[3, 4, 5, 6], default=3, help='Radius for ECFP (default: 3)')
#     parser_ecfp.add_argument('--vector', choices=['count', 'binary'], default='count', help='Type of vector (default: count)')
#     parser_ecfp.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
#     parser_ecfp.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
#                                                           'RRU Dimer', 'RRU Trimer'],
#                                                             default='Monomer', help='polymer unite representation')
#     parser_ecfp.add_argument('--kernel', default=None, help='kernel for GP is optinal')
#     parser_ecfp.add_argument('--transform_type', default='Standard', help='scaler required')


#     # Parser for MACCS model
#     parser_maccs = subparsers.add_parser('maccs', help='Run the MACCS numerical model')
#     parser_maccs.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
#     parser_maccs.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
#     parser_maccs.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
#                                                           'RRU Dimer', 'RRU Trimer'],
#                                                             default='Monomer', help='polymer unite representation')
#     parser_maccs.add_argument('--kernel', default=None, help='kernel for GP is optinal')
#     parser_maccs.add_argument('--transform_type', default='Standard', help='scaler required')

#     # Parser for Mordred model
#     parser_mordred = subparsers.add_parser('mordred', help='Run the Mordred numerical model')
#     parser_mordred.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
#     parser_mordred.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
#     parser_mordred.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
#                                                           'RRU Dimer', 'RRU Trimer'],
#                                                             default='Monomer', help='polymer unite representation')
#     parser_mordred.add_argument('--kernel', default=None, help='kernel for GP is optinal')
#     parser_mordred.add_argument('--transform_type', default='Standard', help='scaler required')

#     # Parse arguments
#     args = parser.parse_args()

#     # Run the appropriate model based on the parsed arguments
#     if args.model == 'ecfp':
#         perform_model_ecfp(args.regressor_type, args.radius, args.vector, args.target, args.oligo_type, args.kernel, args.transform_type)
#     elif args.model == 'maccs':
#         perform_model_maccs(args.regressor_type, args.target, args.oligo_type, args.kernel, args.transform_type)
#     elif args.model == 'mordred':
#         perform_model_mordred(args.regressor_type, args.target, args.oligo_type, args.kernel, args.transform_type)


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
        choices=['RF', 'DT', 'MLR', 'SVR', 'XGBR','KNN', 'GPR', 'NGB', 'sklearn-GPR'], 
        required=True, 
        help="Regressor type required"
    )

    parser.add_argument(
        '--numerical_feats',
        type=str,
        choices=['Mn (g/mol)', 'Mw (g/mol)', 'PDI', 'Temperature SANS/SLS/DLS/SEC (K)',
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

    parser.add_argument(
        '--representation', 
        type=str, 
        choices=['ECFP', 'MACCS', 'Mordred'], 
        required=True, 
        help="Fingerprint required"
    )

    parser.add_argument(
        '--oligomer_representation', 
        type=str, 
        choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer', 'RRU Dimer', 'RRU Trimer'], 
        required=True, 
        help="Fingerprint required"
    )

    parser.add_argument(
        '--radius',
        type=int,
        choices=[3, 4, 5, 6],
        nargs='?',  # This allows the argument to be optional
        default=None,  # Set the default value to None
        help='Radius for ECFP'
    )

    parser.add_argument(
        '--vector',
        type=str,
        choices=['count', 'binary'],
        nargs='?',  # This allows the argument to be optional
        default='count',  # Set the default value to None
        help='Type of vector (default: count)'
    )


    return parser.parse_args()

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
    #     transform_type=args.transform_type,
    #     second_transformer=None,
    #     hyperparameter_optimization=True,
    #     columns_to_impute=args.columns_to_impute,  
    #     special_impute=args.special_impute,
    #     numerical_feats=args.numerical_feats,  
    #     imputer=args.imputer,
    #     cutoff=None,  
    # )











# if __name__ == '__main__':
    # main()
    # main_structural(
    #     dataset=w_data,
    #     representation="MACCS",
    #     # radius=3,
    #     # vector="count",
    #     regressor_type="RF",
    #     target_features=['log First Peak (e-5 place holder)'],  
    #     transform_type=None,
    #     second_transformer=None,
    #     hyperparameter_optimization=True,
    #     oligomer_representation="Monomer",
    # )

