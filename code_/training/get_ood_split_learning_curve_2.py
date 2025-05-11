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
                    process_ood_learning_curve_score,
                    get_incremental_split
                )
from all_factories import optimized_models

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
    print(score)
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
    train_sizes = [len(tv_idxs) for tv_idxs, _ in loco_split_idx.values()]
    min_train_size= min(train_sizes)
    learning_curve_predictions = {}
    learning_curve_scores = {}
    for cluster, (tv_idx,test_idx) in loco_split_idx.items():
        if cluster=='Polar':
            cluster_tv_labels = split_for_training(cluster_labels['Side Chain Cluster'],tv_idx)
        elif cluster in ['Fluorene', 'PPV', 'Thiophene']:
            cluster_tv_labels = split_for_training(cluster_labels['substructure cluster'],tv_idx)
        else:
            cluster_tv_labels = split_for_training(cluster_labels,tv_idx)

        X_tv, y_tv = split_for_training(X, tv_idx), split_for_training(y, tv_idx)
        X_test, y_test = split_for_training(X, test_idx), split_for_training(y, test_idx)
        learning_curve_predictions[f'CO_{cluster}'] = {'y_true': y_test.flatten()}
        train_ratios =[
            .1,.3,.5,.7,
            .9]
        min_ratio_to_compare = min_train_size/len(X_tv)
        if min_ratio_to_compare not in train_ratios:
            train_ratios.append(min_ratio_to_compare)

    
        for train_ratio in train_ratios:
            random_stats_splitter:np.ndarray = random_state_generator(train_ratio)

            # TODO: MAKE both down parallel
            for seed in random_stats_splitter:
                if train_ratio ==1:
                    X_train_OOD,y_train_OOD = X_tv, y_tv
                else:
                    X_train_OOD, _, y_train_OOD, _= train_test_split(X_tv, y_tv, train_size=train_ratio,
                                                              random_state=seed,stratify=cluster_tv_labels,shuffle=True)   
                    

                OOD_regressor = build_regressor(algorithm=model_name,
                                                first_tran=transform_type,
                                                second_tran=second_transformer,
                                                preprocessor=preprocessor)
                uncertainty_preprocessr: Pipeline = preprocessor
                test_scores_OOD, train_scores_OOD, y_test_pred_OOD, y_test_uncertainty_OOD = train_and_predict_ood(OOD_regressor, X_train_OOD, y_train_OOD, X_test, y_test,
                                                                                                    return_train_pred=True, algorithm=model_name,
                                                                                                    manual_preprocessor=uncertainty_preprocessr)
                
                learning_curve_scores.setdefault(f'CO_{cluster}', {}).setdefault(f'ratio_{train_ratio}', {})[f'seed_{seed}'] = (train_scores_OOD, test_scores_OOD)
                learning_curve_predictions.setdefault(f'CO_{cluster}', {}).setdefault(f'ratio_{train_ratio}', {})[f'seed_{seed}'] = {
                    'y_test_pred': y_test_pred_OOD.flatten(),
                    'y_test_uncertainty': y_test_uncertainty_OOD.flatten() if y_test_uncertainty_OOD is not None else None,
                }
            
            
            
            # MAKE both down parallel
            test_ratio = len(y_test)/len(y)
            random_stats_splitter_IID_for_test_set:np.ndarray = random_state_generator(test_ratio)
            for seed in random_stats_splitter:
                for ID_seed in random_stats_splitter_IID_for_test_set:
                    X_tv_IID, X_test_IID, y_tv_IID, y_test_IID = train_test_split(X, y, test_size=len(y_test),
                                                                        random_state=ID_seed,shuffle=True)   

                
                    if train_ratio ==1:
                        X_train_IID,y_train_IID = X_tv_IID, y_train_IID
                    else:
                        X_train_IID, _, y_train_IID, _= train_test_split(X_tv_IID, y_tv_IID, train_size=train_ratio,
                                                                        random_state=seed,shuffle=True)
                
                    IID_regressor = build_regressor(algorithm=model_name,
                                                    first_tran=transform_type,
                                                    second_tran=second_transformer,
                                                    preprocessor=preprocessor)
                    


                    uncertainty_preprocessr: Pipeline = preprocessor
                    test_scores_IID, train_scores_IID, y_test_pred_IID, y_test_uncertainty_IID = train_and_predict_ood(IID_regressor, X_train_IID, y_train_IID, X_test_IID, y_test_IID,
                                                                                                                    return_train_pred=True, algorithm=model_name,
                                                                                                                    manual_preprocessor=uncertainty_preprocessr)
                
                    learning_curve_scores.setdefault(f'IID_{cluster}', {}).setdefault(f'ratio_{train_ratio}', {}).setdefault(f'seed_{seed}', {})[f'test_set_seed_{ID_seed}'] = (train_scores_IID, test_scores_IID)
                    learning_curve_predictions.setdefault(f'IID_{cluster}', {}).setdefault(f'ratio_{train_ratio}', {}).setdefault(f'seed_{seed}', {})[f'test_set_seed_{ID_seed}'] = {
                        'y_test_pred': y_test_pred_OOD.flatten(),
                        'y_test_uncertainty': y_test_uncertainty_OOD.flatten() if y_test_uncertainty_OOD is not None else None,
                    }

        for prefix in ['CO', 'IID']:
            learning_curve_scores[f'{prefix}_{cluster}']['training size'] = len(X_tv)
            learning_curve_predictions[f'{prefix}_{cluster}']['training size'] = len(X_tv)
    return learning_curve_scores, learning_curve_predictions



def build_regressor(algorithm,first_tran,second_tran,preprocessor):
    y_transform = get_target_transformer(first_tran, second_tran)
    model = optimized_models(algorithm)

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
    return regressor


def random_state_generator(train_ratio: float) -> np.ndarray:
    if train_ratio >0.7:
        random_state_list = np.arange(5)
    elif train_ratio >0.5:
        random_state_list = np.arange(7)
    elif train_ratio >0.3:
        random_state_list = np.arange(12)
    elif train_ratio >=0.1:
        random_state_list = np.arange(30)
    else:
        random_state_list = np.arange(50)

    return random_state_list



