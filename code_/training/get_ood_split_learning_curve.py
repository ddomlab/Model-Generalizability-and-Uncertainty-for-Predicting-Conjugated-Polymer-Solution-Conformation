from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
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
from training_utils import get_target_transformer, split_for_training,_optimize_hyperparams,set_globals
from scoring import (
                    cross_validate_regressor,
                    process_scores,
                    train_and_predict_ood,
                    process_ood_scores
                )


from get_ood_split import (StratifiedKFoldWithLabels,
                            get_loco_splits)

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



    cluster_labels :np.ndarray = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    loco_split_idx:Dict[int,tuple[np.ndarray]] = get_loco_splits(cluster_labels)
    cluster_names, counts = np.unique(cluster_labels, return_counts=True)
    train_ratios =[.1, .3, .5, .7,.9]
    min_train_size = len(cluster_labels) - min(counts)

    for cluster, (tv_idx,test_idx) in loco_split_idx.items():
        cluster_tv_labels = split_for_training(cluster_labels,tv_idx)
        X_tv, y_tv = split_for_training(X, tv_idx), split_for_training(y,tv_idx)
        X_test, y_test = split_for_training(X, test_idx), split_for_training(y, test_idx)

        min_ratio_to_compare = min_train_size/len(X_tv)
        if min_ratio_to_compare not in train_ratios:
            train_ratios.append(min_ratio_to_compare)

        for train_ratio in train_ratios:
            if train_ratio >0.3:
                random_state_list = np.arange(4)
            elif train_ratio >0.16:
                random_state_list = np.arange(5)
            elif train_ratio >0.1:
                random_state_list = np.arange(6)

            for seed in random_state_list:
                if train_ratio ==1:
                    X_train,y_train = X_tv, y_tv
                else:
                    X_train, _, y_train, _= train_test_split(X_tv, y_tv, train_size=train_ratio, random_state=seed)   


