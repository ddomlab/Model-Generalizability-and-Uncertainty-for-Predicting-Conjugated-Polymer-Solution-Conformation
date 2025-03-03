from itertools import product
from typing import Callable, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)


from sklearn.metrics._scorer import r2_scorer
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.model_selection import learning_curve
from _validation import multioutput_cross_validate 


def pearson(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    r = pearsonr(y_true, y_pred)[0]
    return r
r_scorer = make_scorer(pearson, greater_is_better=True)


# from training_utils import N_FOLDS, SEEDS


def rmse_score(y_test: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the root mean squared error.

    Args:
        y_test: Test targets.
        y_pred: Predicted targets.

    Returns:
        Root mean squared error.
    """
    return root_mean_squared_error(y_test, y_pred)


# def np_r(y_true: pd.Series, y_pred: np.ndarray) -> float:
#     """
#     Calculate the Pearson correlation coefficient.

#     Args:
#         y_true: Test targets.
#         y_pred: Predicted targets.

#     Returns:
#         Pearson correlation coefficient.
#     """
#     if y_true.type == pd.Series:
#         y_true = y_true.to_numpy().flatten()
#     y_pred = y_pred.tolist()
#     r = np.corrcoef(y_true, y_pred, rowvar=False)[0, 1]
#     return r


def pearson(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    r = pearsonr(y_true, y_pred)[0]
    return r


# r_scorer = make_scorer(r_regression, greater_is_better=True)
# r_scorer = make_scorer(np_r, greater_is_better=True)
# r2_score_multiopt = r2_score(y_true, y_pred, multioutput="raw_values")
def r2_scorer_multi(y_true, y_pred):
    return r2_score(y_true, y_pred, multioutput="raw_values")

def rmse_scorer_multi(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred, multioutput="raw_values")

def mae_scorer_multi(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred, multioutput="raw_values")


def accuracy_scorer_multi(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def f1_scorer_multi(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None) 

def roc_auc_scorer_multi(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average=None)

def recall_scorer_multi(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)

def precision_scorer_multi(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None)


r_scorer = make_scorer(pearson, greater_is_better=True)
rmse_scorer = make_scorer(rmse_score, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

f1_scorer = make_scorer(f1_score, greater_is_better=True)
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)
recall_scorer = make_scorer(recall_score, greater_is_better=True)
precision_scorer = make_scorer(precision_score, greater_is_better=True)


def process_scores(
    scores: dict[int, dict[str, float]],classification,
    ) -> dict[Union[int, str], dict[str, float]]:
        # print(scores)
        # 'test_roc_auc'
        first_key = list(scores.keys())[0]
        score_types: list[str] = [
                key for key in scores[first_key].keys() if key.startswith("test_")
            ]
        if classification:
            #TODO: add multi and single
            print(scores)
            arr = np.array(scores[42]['test_f1'])

            if arr.ndim > 1 and arr.shape[1] > 1:
                avg_f1 = np.round(np.mean(np.vstack([arr for seed in scores.values() for arr in seed["test_f1"]]), axis=0), 3)
                stdev_f1 = np.round(np.std(np.vstack([arr for seed in scores.values() for arr in seed["test_f1"]]), axis=0), 3)
                print("Average scores:\t",
                    # f"r: {avg_r}±{stdev_r}\t",
                    f"f1: {avg_f1}±{stdev_f1}")
                
                avgs: list[float] = [
                        np.mean(np.vstack([arr for seed in scores.values() for arr in seed[score]]), axis=0) for score in score_types
                    ]
                stdevs: list[float] = [
                        np.std(np.vstack([arr for seed in scores.values() for arr in seed[score]]), axis=0) for score in score_types
                    ]
                
            else:
                avg_f1 = round(np.mean([seed["test_f1"] for seed in scores.values()]), 2)
                stdev_f1 = round(np.std([seed["test_f1"] for seed in scores.values()]), 2)
                print("Average scores:\t",
                    # f"r: {avg_r}±{stdev_r}\t",
                    f"f1: {avg_f1}±{stdev_f1}")


                avgs: list[float] = [
                    np.mean([seed[score] for seed in scores.values()]) for score in score_types
                ]
                stdevs: list[float] = [
                    np.std([seed[score] for seed in scores.values()]) for score in score_types
                ]
        else:

            arr = np.array(scores[42]["test_r2"])

            if arr.ndim > 1 and arr.shape[1] > 1:

                avg_r2 = np.round(np.mean(np.vstack([arr for seed in scores.values() for arr in seed["test_r2"]]), axis=0), 3)
                stdev_r2 = np.round(np.std(np.vstack([arr for seed in scores.values() for arr in seed["test_r2"]]), axis=0), 3)
                print("Average scores:\t",
                    # f"r: {avg_r}±{stdev_r}\t",
                    f"r2: {avg_r2}±{stdev_r2}")
                
                avgs: list[float] = [
                    np.mean(np.vstack([arr for seed in scores.values() for arr in seed[score]]), axis=0) for score in score_types
                ]
                stdevs: list[float] = [
                    np.std(np.vstack([arr for seed in scores.values() for arr in seed[score]]), axis=0) for score in score_types
                ]
                # print(avgs)
            else:
                avg_r = round(np.mean([seed["test_r"] for seed in scores.values()]), 2)
                stdev_r = round(np.std([seed["test_r"] for seed in scores.values()]), 2)
                avg_r2 = round(np.mean([seed["test_r2"] for seed in scores.values()]), 2)
                stdev_r2 = round(np.std([seed["test_r2"] for seed in scores.values()]), 2)
                print("Average scores:\t",
                    f"r: {avg_r}±{stdev_r}\t",
                    f"r2: {avg_r2}±{stdev_r2}")


                avgs: list[float] = [
                    np.mean([seed[score] for seed in scores.values()]) for score in score_types
                ]
                stdevs: list[float] = [
                    np.std([seed[score] for seed in scores.values()]) for score in score_types
                ]


        score_types: list[str] = [score.replace("test_", "") for score in score_types]
        for score, avg, stdev in zip(score_types, avgs, stdevs ):
            scores[f"{score}_avg"] = abs(avg) if score in ["rmse", "mae"] else avg
            scores[f"{score}_stdev"] = stdev
        
        if arr.ndim > 1 and arr.shape[1] > 1:
            for score in score_types:
                scores[f"{score}_avg_aggregate"] = np.mean(scores[f"{score}_avg"])
                scores[f"{score}_stdev_aggregate"] = np.mean(scores[f"{score}_stdev"])
        # print(scores)
        return scores


def process_learning_score(score: dict[int, dict[str, np.ndarray]]):
     # Initialize arrays for aggregation
    train_scores_mean = None
    train_scores_std = None
    test_scores_mean = None
    test_scores_std = None

    # Loop over seeds and accumulate results
    for _, results in score.items():
        if train_scores_mean is None:
            # Initialize mean and std with the first seed's results
            train_scores_mean = results["train_scores"].mean(axis=1, keepdims=True)
            train_scores_std = results["train_scores"].std(axis=1, keepdims=True)
            test_scores_mean = results["test_scores"].mean(axis=1, keepdims=True)
            test_scores_std = results["test_scores"].std(axis=1, keepdims=True)
        else:
            # Accumulate the means and stds
            train_scores_mean += results["train_scores"].mean(axis=1, keepdims=True)
            train_scores_std += results["train_scores"].std(axis=1, keepdims=True)
            test_scores_mean += results["test_scores"].mean(axis=1, keepdims=True)
            test_scores_std += results["test_scores"].std(axis=1, keepdims=True)

    # Calculate the averages over the number of seeds
    num_seeds = len(score)
    score['aggregated_results']= {
        "train_sizes": results["train_sizes"],
        "train_sizes_fraction": results["train_sizes_fraction"],
        "train_scores_mean": train_scores_mean / num_seeds,
        "train_scores_std": train_scores_std / num_seeds,
        "test_scores_mean": test_scores_mean / num_seeds,
        "test_scores_std": test_scores_std / num_seeds,
    }

    return score



def cross_validate_regressor(
    regressor, X, y, cv,classification=False,
    ) -> tuple[dict[str, float], np.ndarray]:

  
        if y.shape[1]>1:
            if classification:
                scorers= {
                "accuracy": accuracy_scorer_multi,  
                "f1": f1_scorer_multi,
                "recall": recall_scorer_multi,
                "precision": precision_scorer_multi,
                "roc_auc": roc_auc_scorer_multi    
                }
            else:
                scorers = {
                        "r2": r2_scorer_multi,
                        "rmse": rmse_scorer_multi,
                        "mae": mae_scorer_multi
                        }
        

            score =  multioutput_cross_validate(
                estimator= regressor,
                X=X,
                y= y,
                cv=cv,
                scorers=scorers,
                n_jobs=-1,
                verbose=0
                )

        else:
            if classification:
                scorers= {
                # "accuracy": accuracy_scorer_multi,  
                "f1": f1_scorer,
                "recall": recall_scorer,
                "precision": precision_scorer,
                "roc_auc": roc_auc_scorer    
                }
            else:
                scorers = {
                    #r pearson is added
                    "r": r_scorer,
                    "r2": r2_scorer,
                    "rmse": rmse_scorer,
                    "mae": mae_scorer,
                }
            score: dict[str, float] = cross_validate(
                regressor,
                X,
                y,
                cv=cv,
                scoring=scorers,
                # return_estimator=True,
                n_jobs=-1,
                )

        predictions: np.ndarray = cross_val_predict(
            regressor,
            X,
            y,
            cv=cv,
            n_jobs=-1,
        )
        return score, predictions


def get_incremental_split(
        regressor_params, X, y, cv, steps:int,
        random_state:int
    ) -> tuple:
     
    training_sizes, training_scores, testing_scores = learning_curve(
                                                        regressor_params,
                                                        X,
                                                        y,
                                                        cv=cv,
                                                        n_jobs=-1,
                                                        train_sizes=np.linspace(0.1, 1, int(0.9 / steps)),
                                                        scoring="r2",
                                                        shuffle=True,
                                                        random_state=random_state
                                                        )

 
    return training_sizes, training_scores, testing_scores


# def get_score_func(score: str, output: str) -> Callable:
#     """
#     Returns the appropriate scoring function for the given output.
#     """
#     score_func: Callable = score_lookup[score][output]
#     return score_func



def train_and_predict_ood(regressor, X_train_val, y_train_val, X_test, y_test, seed):
    regressor.fit(X_train_val, y_train_val)
    
    y_pred = regressor.predict(X_test)
    scores = get_prediction_scores(y_test, y_pred)
    
    return scores, y_pred


def get_prediction_scores(y_test, y_pred):
    return {
        "test_mad": (y_test - y_test.mean()).abs().mean(),
        "test_std": y_test.std(),
        "test_mae": mae_scorer(y_test, y_pred),
        "test_rmse": rmse_scorer(y_test, y_pred),
        "test_r2": r2_score(y_test, y_pred),
        "test_pearson_r": pearsonr(y_test, y_pred)[0],
        "test_pearson_p_value": pearsonr(y_test, y_pred)[1],
        "test_spearman_r": spearmanr(y_test, y_pred)[0],
        "test_spearman_p_value": spearmanr(y_test, y_pred)[1],
        "test_kendall_r": kendalltau(y_test, y_pred)[0],
        "test_kendall_p_value": kendalltau(y_test, y_pred)[1],
    }