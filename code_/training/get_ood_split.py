from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV
from data_handling import remove_unserializable_keys, save_results
from filter_data import filter_dataset
from all_factories import (
                            regressor_factory,
                            # regressor_search_space,
                            transforms,
                            construct_kernel,
                            get_regressor_search_space)


from imputation_normalization import preprocessing_workflow
from scoring import (
    cross_validate_regressor,
    process_scores,
)


def set_globals(Test: bool=False) -> None:
    global SEEDS, N_FOLDS, BO_ITER
    if not Test:
        SEEDS = [6, 13, 42, 69, 420, 1234567890, 473129]
        N_FOLDS = 5
        BO_ITER = 42
    else:
        SEEDS = [42,13]
        N_FOLDS = 2
        BO_ITER = 1


def get_cluster_splits(df, cluster_type:str):
    clusters = df[cluster_type].to_numpy()
    cluster_names = np.unique(clusters)
    n_clusters =  len(np.unique(clusters))
    splits = {}

    if n_clusters>2:
        # use stratified
        for  n in cluster_names:
            mask = 
            # splits[name] = (train, test)
    else:
        # use CV
        mask = # the one which is rare
        # splits[name] = (train, test)
    return splits