import pandas as pd
from pathlib import Path
from train_generalizability_utils import get_generalizability_predictions
from all_factories import radius_to_bits,cutoffs
import sys
import json
import numpy as np
sys.path.append("../cleaning")
from argparse import ArgumentParser
from data_handling import save_results


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped.pkl"
w_data = pd.read_pickle(training_df_dir)
# ['Monomer', 'Dimer', 'Trimer', 'RRU Monomer', 'RRU Dimer', 'RRU Trimer']
TEST=False



def main_mordred_numerical(
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
    
    columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    special_column: str = "Mw (g/mol)"
    numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI",
                                   "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)",
                                   'solvent dD', 'solvent dH', 'solvent dP']
                              

    imputer = "mean"
    learning_score  = get_generalizability_predictions(
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
    save_results(scores=None,
                predictions=None,
                imputer=imputer,
                generalizability_score=learning_score,
                df_shapes=None,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                cutoff=cutoffs,
                TEST=TEST,
                hypop=hyperparameter_optimization
                )


def perform_model_mordred_numerical(regressor_type:str,target:str,oligo_type:str):
            print(f'polymer representation: {oligo_type}')
            main_mordred_numerical(dataset=w_data,
                                    regressor_type=regressor_type,
                                    transform_type= "Standard",
                                    hyperparameter_optimization= False,
                                    target_features= [target],
                                    oligomer_representation=oligo_type
                                    )


# perform_model_mordred_numerical('RF','Rg1 (nm)')



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
    numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI",
                                   "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)",
                                   'solvent dD', 'solvent dH', 'solvent dP']
    imputer = "mean"

    learning_score  = get_generalizability_predictions(
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
    save_results(scores=None,
                predictions=None,
                imputer=imputer,
                generalizability_score=learning_score,
                df_shapes=None,
                representation= representation,
                pu_type= oligomer_representation,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                cutoff=cutoffs,
                TEST=TEST,
                hypop=hyperparameter_optimization
                )



def perform_model_maccs_numerical(regressor_type:str,target:str,oligo_type:str):
            print(f'polymer representation: {oligo_type}')
            main_maccs_numerical(dataset=w_data,
                                regressor_type=regressor_type,
                                transform_type= "Standard",
                                hyperparameter_optimization= True,
                                target_features= [target],
                                oligomer_representation=oligo_type
                                )




# perform_model_maccs_numerical('RF','Rg1 (nm)')


def main_ecfp_numerical(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    radius: int,
    oligomer_representation: str,
    vector_type: str,
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

    columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    special_column: str = "Mw (g/mol)"
    numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI",
                                   "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)",
                                   'solvent dD', 'solvent dH', 'solvent dP']

    imputer = "mean"
    learning_score  = get_generalizability_predictions(
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
    save_results(scores=None,
                predictions=None,
                imputer=imputer,
                generalizability_score=learning_score,
                df_shapes=None,
                representation= representation,
                pu_type= oligomer_representation,
                radius= radius,
                vector =vector_type,
                target_features=target_features,
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                cutoff=cutoffs,
                TEST=TEST,
                hypop=hyperparameter_optimization
                )



def perform_model_ecfp(regressor_type:str, radius:int,vector:str,target:str,oligo_type:str):
                print(f'polymer representation: {oligo_type}')
                main_ecfp_numerical(
                                    dataset=w_data,
                                    regressor_type= regressor_type,
                                    target_features= [target],
                                    transform_type= "Standard",
                                    hyperparameter_optimization= True,
                                    radius = radius,
                                    vector_type=vector,
                                    oligomer_representation=oligo_type,
                                    )



def main():
    parser = ArgumentParser(description='Run models with specific parameters')

    # Subparsers for different models
    subparsers = parser.add_subparsers(dest='model', required=True, help='Choose a model to run')

    # Parser for ECFP model
    parser_ecfp = subparsers.add_parser('ecfp', help='Run the ECFP model')
    parser_ecfp.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
    parser_ecfp.add_argument('--radius', type=int, choices=[3, 4, 5, 6], default=3, help='Radius for ECFP (default: 3)')
    parser_ecfp.add_argument('--vector', choices=['count', 'binary'], default='count', help='Type of vector (default: count)')
    parser_ecfp.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
    parser_ecfp.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
                                                          'RRU Dimer', 'RRU Trimer'],
                                                            default='Monomer', help='polymer unite representation')
    # Parser for MACCS model
    parser_maccs = subparsers.add_parser('maccs', help='Run the MACCS numerical model')
    parser_maccs.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
    parser_maccs.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
    parser_maccs.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
                                                          'RRU Dimer', 'RRU Trimer'],
                                                            default='Monomer', help='polymer unite representation')
    # Parser for Mordred model
    parser_mordred = subparsers.add_parser('mordred', help='Run the Mordred numerical model')
    parser_mordred.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
    parser_mordred.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
    parser_mordred.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
                                                          'RRU Dimer', 'RRU Trimer'],
                                                            default='Monomer', help='polymer unite representation')
    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate model based on the parsed arguments
    if args.model == 'ecfp':
        perform_model_ecfp(args.regressor_type, args.radius, args.vector, args.target, args.oligo_type)
    elif args.model == 'maccs':
        perform_model_maccs_numerical(args.regressor_type, args.target, args.oligo_type)
    elif args.model == 'mordred':
        perform_model_mordred_numerical(args.regressor_type, args.target, args.oligo_type)

if __name__ == '__main__':
    main()
    # perform_model_mordred_numerical('NGB','Rg1 (nm)', 'Monomer')







# perform_model_ecfp(regressor_type='RF', radius=3,vector='count',target='Rg1 (nm)')

# perform_model_maccs_numerical('RF','Rg1 (nm)')

# main_numerical_only(
#     dataset=w_data,
#     regressor_type="DT",
#     target_features=["Rg1 (nm)"],  # Can adjust based on actual usage
#     transform_type="Standard",
#     hyperparameter_optimization=True,
#     columns_to_impute=["PDI"],
#     special_impute="Mw (g/mol)",
#     numerical_feats=["Mn (g/mol)", "Mw (g/mol)", "PDI"],
#     imputer='mean',
#     cutoff=None)

    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]


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



