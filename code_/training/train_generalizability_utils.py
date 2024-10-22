import json
import platform
from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from skopt import BayesSearchCV
# from skorch.regressor import NeuralNetRegressor
# from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import time

# from data_handling import remove_unserializable_keys, save_results
from filter_data import filter_dataset
from all_factories import (
                            regressor_factory,
                            regressor_search_space,
                            transforms)

from training_utils import (
     _optimize_hyperparams,
     split_for_training,

)


from imputation_normalization import preprocessing_workflow
from scoring import (
    # cross_validate_regressor,
    process_learning_score,
    get_incremental_split
)

from scipy.stats import pearsonr

# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score,
# )


HERE: Path = Path(__file__).resolve().parent

# os_type: str = platform.system().lower()
TEST=False

# Seeds for generating random states
if TEST==False:
    SEEDS = [6, 13, 42, 69, 420, 1234567890, 473129]
else:    
    SEEDS = [42]
N_FOLDS: int = 5 if not TEST else 2

# Number of iterations for Bayesian optimization
BO_ITER: int = 42 if not TEST else 1

# Number of folds for cross-validation
# N_FOLDS: int = 5 if not TEST else 2

# # Number of iterations for Bayesian optimization
# BO_ITER: int = 42 if not TEST else 1

# Path to config for Pytorch model
# CONFIG_PATH: Path = HERE / "ANN_config.json"

# Set seed for PyTorch model
# torch.manual_seed(0)

def get_generalizability_predictions(
    dataset: pd.DataFrame,
    features_impute: Optional[list[str]],
    special_impute: Optional[str],
    representation: Optional[str],
    structural_features: Optional[list[str]],
    numerical_feats: Optional[list[str]],
    unroll: Union[dict[str, str], list[dict[str, str]], None],
    regressor_type: str,
    target_features: str,
    transform_type: str,
    hyperparameter_optimization: bool=True,
    imputer: Optional[str] = None,
    cutoff:Dict[str, Tuple[Optional[float], Optional[float]]]=None,
    # output_dir_name: str = "results",
    ) -> None:
        """
        you should change the name here for prepare
        """
            #seed scores and seed prediction
        learning_score = _prepare_data(
                                            dataset=dataset,
                                            features_impute= features_impute,
                                            special_impute= special_impute,
                                            representation=representation,
                                            structural_features=structural_features,
                                            unroll=unroll,
                                            numerical_feats = numerical_feats,
                                            target_features=target_features,
                                            regressor_type=regressor_type,
                                            transform_type=transform_type,
                                            imputer=imputer,
                                            cutoff=cutoff,
                                            hyperparameter_optimization=hyperparameter_optimization,
                                            )
        
        
        scores = process_learning_score(learning_score)
        return scores
        



def _prepare_data(
    dataset: pd.DataFrame,
    target_features: str,
    regressor_type: str,
    features_impute: Optional[list[str]]=None,
    special_impute: Optional[str]=None,
    representation: Optional[str]=None,
    structural_features: Optional[list[str]]=None,
    numerical_feats: Optional[list[str]]=None,
    unroll: Union[dict, list, None] = None,
    transform_type: str = "Standard",
    hyperparameter_optimization: bool = True,
    imputer: Optional[str] = None,
    cutoff: Dict[str, Tuple[Optional[float], Optional[float]]]=None,
    **kwargs,
    ) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:


    """
    here you should change the names
    """



    X, y, unrolled_feats, _ = filter_dataset(
                                        raw_dataset=dataset,
                                        structure_feats=structural_features,
                                        scalar_feats=numerical_feats,
                                        target_feats=target_features,
                                        cutoff=cutoff,
                                        dropna = True,
                                        unroll=unroll,
                                        )

    # Pipline workflow here and preprocessor
    preprocessor: Pipeline = preprocessing_workflow(imputer=imputer,
                                                    feat_to_impute=features_impute,
                                                    representation = representation,
                                                    numerical_feat=numerical_feats,
                                                    structural_feat = unrolled_feats,
                                                    special_column=special_impute,
                                                    scaler=transform_type)
    
    preprocessor.set_output(transform="pandas")
    learning_score = run_leaning(
                                    X,
                                    y,
                                    preprocessor=preprocessor,
                                    regressor_type=regressor_type,
                                    transform_type=transform_type,
                                    hyperparameter_optimization=hyperparameter_optimization,
                                    **kwargs,
                                    )

    return learning_score

def run_leaning(
    X, y, preprocessor: Union[ColumnTransformer, Pipeline], regressor_type: str,
    transform_type: str, hyperparameter_optimization: bool = True,
    **kwargs,
    ) -> dict[int, dict[str, float]]:

    seed_learning_curve_scores:dict[int, dict] = {}

    for seed in SEEDS:
        cv_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        y_transform = Pipeline(
                    steps=[("y scaler",transforms[transform_type]),
                        ])

        y_transform_regressor: TransformedTargetRegressor = TransformedTargetRegressor(
                regressor=regressor_factory[regressor_type],
                transformer=y_transform,
                )
        #   print('yes')
        regressor :Pipeline= Pipeline(steps=[
                        ("preprocessor", preprocessor),
                        ("regressor", y_transform_regressor),
                            ])

        # set_output on dataframe
        regressor.set_output(transform="pandas")
        if hyperparameter_optimization:
                best_estimator, regressor_params = _optimize_hyperparams(
                    X,
                    y,
                    cv_outer=cv_outer,
                    seed=seed,
                    regressor_type=regressor_type,
                    regressor=regressor,
                )
        
                train_sizes, train_scores, test_scores = get_incremental_split(best_estimator,
                                                                                X,
                                                                                y,
                                                                                cv_outer,
                                                                                steps=0.1,
                                                                                random_state=seed)


        elif regressor_type == "ANN":
                pass

        else:
                train_sizes, train_scores, test_scores = get_incremental_split(regressor,
                                                                                X,
                                                                                y,
                                                                                cv_outer,
                                                                                steps=0.1,
                                                                                random_state=seed)


        seed_learning_curve_scores[seed] = {
                    "train_sizes": train_sizes,   # 1D array of training sizes used
                    "train_sizes_fraction": train_sizes/len(X),
                    "train_scores": train_scores,  # 2D array of training scores
                    "test_scores": test_scores,  # 2D array of validation (cross-validation) scores
                    "best_params": regressor_params
                }

    return seed_learning_curve_scores






