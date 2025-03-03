from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple
from sklearn.pipeline import Pipeline
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


def get_loco_splits(X,y, cluster_type:np.ndarray):
    cluster_names, counts = np.unique(cluster_type, return_counts=True)
    n_clusters = len(cluster_names)
    splits = {}
    if n_clusters>2:
        # use stratified
        for  n in cluster_names:
            mask = cluster_type == n
            test_indices = np.where(mask)[0]
            train_indices = np.where(np.logical_not(mask))[0]
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            splits[n] = (X_train, y_train, X_test, y_test)
    else:
        rare_cluster = cluster_names[np.argmin(counts)]  # Identify the rare cluster
        mask = cluster_type == rare_cluster
        test_indices = np.where(mask)[0]
        train_indices = np.where(np.logical_not(mask))[0]
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        splits[rare_cluster] = (X_train, y_train, X_test, y_test)
    return splits


def get_optimization_split(X_tr,y_tr,n_cluster,n_split_size,seed):
    if n_cluster>2:
        
# fit.(x_train,y_train)
# predict.(x_test,y_test)

# for cluster, (X_train, y_train, X_test, y_test) in splits.items():
    # optimize(x_train,y_train,n_splits=len(splits),seed)
    # fit.(x_train,y_train)
    # predict.(x_test,y_test)

def optimize(regressor, search_space, scoring, x_train,y_train,cluster_lables:np.ndarray, n_split_size, seed):
    estimators: list[BayesSearchCV] = []
    if n_split_size>2:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, val_index in cv.split(x_train, cluster_lables):
                X_tv = x_train[train_index]
                y_tv = y_train[train_index]  

                bayes = BayesSearchCV(
                regressor,
                search_space,
                n_iter=BO_ITER,
                cv=None,
                n_jobs=-1,
                random_state=seed,
                refit=True,
                scoring=scoring,
                return_train_score=True,
                )
                bayes.fit(X_tv, y_tv)
                
                print(f"\n\nBest parameters: {bayes.best_params_}\n\n")
                estimators.append(bayes)
    else: 
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, val_index in cv.split(x_train, y_train):
            X_tv = x_train[train_index]
            y_tv = y_train[train_index]     
            bayes = BayesSearchCV(
                    regressor,
                    search_space,
                    n_iter=BO_ITER,
                    cv=None,
                    n_jobs=-1,
                    random_state=seed,
                    refit=True,
                    scoring=scoring,
                    return_train_score=True,
                    )
            
            bayes.fit(X_tv, y_tv)
            print(f"\n\nBest parameters: {bayes.best_params_}\n\n")
            estimators.append(bayes)

    best_idx: int = np.argmax([est.best_score_ for est in estimators])
    best_estimator: Pipeline = estimators[best_idx].best_estimator_
    try:
        regressor_params: dict = best_estimator.named_steps.regressor.get_params()
        regressor_params = remove_unserializable_keys(regressor_params)
    except:
        regressor_params = {"bad params": "couldn't get them"}

    return best_estimator, regressor_params

