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
                            transforms,
                            get_regressor_search_space)

from imputation_normalization import preprocessing_workflow
from training_utils import get_target_transformer, split_for_training,set_globals
from scoring import (
                    train_and_predict_ood,
                    process_ood_learning_curve_score
                )
from all_factories import optimized_models

from get_ood_split import (StratifiedKFoldWithLabels,
                            get_loco_splits)

def set_globals(Test: bool=False) -> None:
    global SEEDS, N_FOLDS, BO_ITER
    if not Test:
        SEEDS = [6, 13, 42, 69, 420, 1234567890]
        N_FOLDS = 5
        BO_ITER = 42
    else:
        SEEDS = [42,13]
        N_FOLDS = 2
        BO_ITER = 1





def train_ood_learning_curve(
                            dataset: pd.DataFrame,
                            target_features: str,
                            representation: Optional[str],
                            structural_features: Optional[list[str]],
                            numerical_feats: Optional[list[str]],
                            unroll: Union[dict[str, str], list[dict[str, str]], None],
                            regressor_type: str,
                            transform_type: str=None,
                            second_transformer:str=None,
                            clustering_method:str= None,
                            Test:bool=False,
                            ):
                          

    set_globals(Test)
    score, predictions = prepare_data(
                                    dataset=dataset,
                                    target_features=target_features,
                                    regressor_type=regressor_type,
                                    representation=representation,
                                    structural_features=structural_features,
                                    numerical_feats=numerical_feats,
                                    unroll=unroll,
                                    transform_type=transform_type,
                                    second_transformer=second_transformer,
                                    clustering_method=clustering_method,
                                    )
    print(score)

    score = process_ood_learning_curve_score(score)

    return score, predictions

def prepare_data(    
                dataset: pd.DataFrame,
                target_features: str,
                regressor_type: str,
                representation: Optional[str]=None,
                structural_features: Optional[list[str]]=None,
                numerical_feats: Optional[list[str]]=None,
                unroll: Union[dict, list, None] = None,
                transform_type: str = None,
                second_transformer:str = None,
                clustering_method:str= None,
                ):

    X, y, unrolled_feats, cluster_labels, _ = filter_dataset(
                                                    raw_dataset=dataset,
                                                    structure_feats=structural_features,
                                                    scalar_feats=numerical_feats,
                                                    target_feats=target_features,
                                                    dropna = True,
                                                    unroll=unroll,
                                                    cluster_type=clustering_method,
                                                    )

    preprocessor: Pipeline = preprocessing_workflow(imputer=None,
                                                    feat_to_impute=None,
                                                    representation = representation,
                                                    numerical_feat=numerical_feats,
                                                    structural_feat = unrolled_feats,
                                                    special_column=None,
                                                    scaler=transform_type
                                                    )
    preprocessor.set_output(transform="pandas")

    return run_ood_learning_curve(X, y,
                                    cluster_labels,
                                    regressor_type,
                                    transform_type,
                                    second_transformer,
                                    preprocessor
                                    )




def run_ood_learning_curve(
                            X:np.ndarray,
                            y:np.ndarray,
                            cluster_labels:Union[np.ndarray, dict[str, np.ndarray]],
                            model_name:str,
                            transform_type:str = None,
                            second_transformer:str = None,
                            preprocessor: Optional[Pipeline]=None,
                            )-> Tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
                          



    loco_split_idx:Dict[int,tuple[np.ndarray]] = get_loco_splits(cluster_labels)
    # cluster_names, counts = np.unique(cluster_labels, return_counts=True)
    train_ratios =[.1, .3, .5, .7, .9]
    train_sizes = [len(tv_idxs) for tv_idxs, _ in loco_split_idx.values()]
    min_train_size= min(train_sizes)
    learning_curve_predictions = {}
    learning_curve_scores = {}
    print(loco_split_idx)
    for cluster, (tv_idx,test_idx) in loco_split_idx.items():
        if cluster=='ionic-EG':
            cluster_tv_labels = split_for_training(cluster_labels['EG-Ionic-Based Cluster'],tv_idx)
        elif cluster in ['Fluorene', 'PPV', 'Thiophene']:
            cluster_tv_labels = split_for_training(cluster_labels['substructure cluster'],tv_idx)
        else:
            cluster_tv_labels = split_for_training(cluster_labels,tv_idx)

        X_tv, y_tv = split_for_training(X, tv_idx), split_for_training(y, tv_idx)
        X_test, y_test = split_for_training(X, test_idx), split_for_training(y, test_idx)

        min_ratio_to_compare = min_train_size/len(X_tv)
        if min_ratio_to_compare not in train_ratios:
            train_ratios.append(min_ratio_to_compare)

        for train_ratio in train_ratios:
            if train_ratio >0.7:
                random_state_list = np.arange(3)
            elif train_ratio >0.5:
                random_state_list = np.arange(4)
            elif train_ratio >0.3:
                random_state_list = np.arange(6)
            elif train_ratio >=0.1:
                random_state_list = np.arange(10)

            else:
                random_state_list = np.arange(15)

            for seed in random_state_list:
                if train_ratio ==1:
                    X_train,y_train = X_tv, y_tv
                else:
                    X_train, _, y_train, _= train_test_split(X_tv, y_tv, train_size=train_ratio,
                                                              random_state=seed,stratify=cluster_tv_labels)   
                



                y_transform = get_target_transformer(transform_type, second_transformer)
                model = optimized_models(model_name, random_state=seed)

                y_model = TransformedTargetRegressor(
                                    regressor=model,
                                    transformer=y_transform,
                            )
                new_preprocessor = 'passthrough' if len(preprocessor.steps) == 0 else preprocessor
                regressor :Pipeline= Pipeline(steps=[
                                    ("preprocessor", new_preprocessor),
                                    ("regressor", y_model),
                                        ])
                regressor.set_output(transform="pandas")

                train_scores, test_score, y_test_pred_ood = train_and_predict_ood(regressor,X_train, y_train,X_test, y_test,return_train_pred=True)
                learning_curve_scores.setdefault(f'CO_{cluster}', {}).setdefault(f'ratio_{train_ratio}', {})[f'seed_{seed}'] = (train_scores, test_score)
                learning_curve_predictions.setdefault(f'CO_{cluster}', {}).setdefault(f'ratio_{train_ratio}', {})[f'seed_{seed}'] = {
                    'y_true': y_test,
                    'y_test_pred': y_test_pred_ood
                }
            # print(learning_curve_scores)
    return learning_curve_scores, learning_curve_predictions



