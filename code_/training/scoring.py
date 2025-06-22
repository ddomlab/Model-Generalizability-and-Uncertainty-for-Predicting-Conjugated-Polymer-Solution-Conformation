from itertools import product
from typing import Callable, Union, Dict, List
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor
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

from sklearn.preprocessing import StandardScaler
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


# def pearson(y_true: pd.Series, y_pred: np.ndarray) -> float:
#     if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
#         y_true = y_true.to_numpy()
#     y_true = y_true.flatten()
#     y_pred = y_pred.flatten()
#     r = pearsonr(y_true, y_pred)[0]
#     return r

def flatten_array(arr):
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.to_numpy()
    return arr.flatten()

def pearson_r(y_true, y_pred):
    y_true, y_pred = flatten_array(y_true), flatten_array(y_pred)
    return pearsonr(y_true, y_pred)[0]

def pearson_p(y_true, y_pred):
    y_true, y_pred = flatten_array(y_true), flatten_array(y_pred)
    return pearsonr(y_true, y_pred)[1]

def spearman_r(y_true, y_pred):
    y_true, y_pred = flatten_array(y_true), flatten_array(y_pred)
    return spearmanr(y_true, y_pred)[0]

def spearman_p(y_true, y_pred):
    y_true, y_pred = flatten_array(y_true), flatten_array(y_pred)
    return spearmanr(y_true, y_pred)[1]

def kendall_r(y_true, y_pred):
    y_true, y_pred = flatten_array(y_true), flatten_array(y_pred)
    return kendalltau(y_true, y_pred)[0]

def kendall_p(y_true, y_pred):
    y_true, y_pred = flatten_array(y_true), flatten_array(y_pred)
    return kendalltau(y_true, y_pred)[1]



# multiouput regression scorers and classification scorers
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


r_scorer = make_scorer(pearson_r, greater_is_better=True)
spearman_scorer = make_scorer(spearman_r, greater_is_better=True)
kendall_scorer = make_scorer(kendall_r, greater_is_better=True)
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score, greater_is_better=True)


# P-value scorers (not typically used for optimization but useful for logging)
pearson_p_scorer = make_scorer(pearson_p, greater_is_better=False)
spearman_p_scorer = make_scorer(spearman_p, greater_is_better=False)
kendall_p_scorer = make_scorer(kendall_p, greater_is_better=False)



f1_scorer = make_scorer(f1_score, greater_is_better=True)
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)
recall_scorer = make_scorer(recall_score, greater_is_better=True)
precision_scorer = make_scorer(precision_score, greater_is_better=True)


def process_scores(
    scores: dict[int, dict[str, float]],classification:bool=False,
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
                avg_r = round(np.mean([seed["test_pearson_r"] for seed in scores.values()]), 2)
                stdev_r = round(np.std([seed["test_pearson_r"] for seed in scores.values()]), 2)
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


def compute_summary_stats(metrics: Dict[str, list[float]]) -> Dict[str, float]:
    """
    Helper function to compute the mean and standard deviation of the test/train metrics.
    """
    summary_stats = {}
    
    for metric, values in metrics.items():
        values = np.array(values, dtype=np.float64)  # Convert to NumPy array for handling NaNs
        valid_values = values[~np.isnan(values)]  # Remove NaNs before computing stats
        
        if valid_values.size > 0:
            # Use absolute mean for RMSE and MAE
            summary_stats[f"{metric}_mean"] = abs(np.mean(valid_values)) if metric in ["test_rmse", "train_rmse", "test_mae", "train_mae"] else np.mean(valid_values)
            summary_stats[f"{metric}_std"] = np.std(valid_values)
        else:
            summary_stats[f"{metric}_mean"] = np.nan
            summary_stats[f"{metric}_std"] = np.nan

    return summary_stats


def process_ood_scores(
    scores: Dict[int, Dict[str, Union[float, list]]]
) -> Dict[Union[int, str], Dict[str, float]]:
    for cluster, seed_data in scores.items():
        if cluster.startswith("CO_") or cluster.startswith("ID_"):
            test_metrics = {}

            # Collect test scores across seeds
            for seed, metrics in seed_data.items():
                if isinstance(seed, int):
                    for key, value in metrics.items():
                        if key.startswith("test_") and not isinstance(value, dict):
                            if isinstance(value, list):
                                test_metrics.setdefault(key, []).extend(value)
                            else:
                                test_metrics.setdefault(key, []).append(value)

            # Compute summary statistics for the cluster
            summary_stats = compute_summary_stats(test_metrics)
            
            # Store the results back in the scores dictionary
            scores[cluster]['summary_stats'] = summary_stats

    return scores


from collections import defaultdict

def process_ood_learning_curve_score(scores: dict) -> dict:
    """
    Process learning curve results for both CO_ and IID_ clusters:
    - CO_: aggregate across seeds
    - IID_: aggregate across seeds and test_set_seeds
    """
    for cluster, score_dict in scores.items():
        if cluster.startswith("CO_"):
            for train_ratio, seeds in score_dict.items():
                if train_ratio == "training size":
                    continue

                test_metrics = defaultdict(list)
                train_metrics = defaultdict(list)

                for seed, (train_results, test_results) in seeds.items():
                    for metric, value in test_results.items():
                        value = value.item() if isinstance(value, np.ndarray) else value
                        test_metrics[metric].append(value)
                    for metric, value in train_results.items():
                        value = value.item() if isinstance(value, np.ndarray) else value
                        train_metrics[metric].append(value)

                scores[cluster][train_ratio].update({
                    "test_summary_stats": compute_summary_stats(test_metrics),
                    "train_summary_stats": compute_summary_stats(train_metrics),
                })

        elif cluster.startswith("IID_"):
            for train_ratio, seeds_dict in score_dict.items():
                if train_ratio == "training size":
                    continue

                test_metrics = defaultdict(list)
                train_metrics = defaultdict(list)

                for seed, test_seed_dict in seeds_dict.items():
                    for test_seed, (train_results, test_results) in test_seed_dict.items():
                        for metric, value in test_results.items():
                            value = value.item() if isinstance(value, np.ndarray) else value
                            test_metrics[metric].append(value)
                        for metric, value in train_results.items():
                            value = value.item() if isinstance(value, np.ndarray) else value
                            train_metrics[metric].append(value)

                scores[cluster][train_ratio].update({
                    "test_summary_stats": compute_summary_stats(test_metrics),
                    "train_summary_stats": compute_summary_stats(train_metrics),
                })

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

        # MULTIOUPUT 
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

        # SINGLE OUTPUT
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
                    "pearson_r": r_scorer,
                    # "pearson_p": pearson_p_scorer,
                    "spearman_r": spearman_scorer,
                    # "spearman_p": spearman_p_scorer,
                    # "kendall_r": kendall_scorer,
                    # "kendall_p": kendall_p_scorer,
                    "rmse": rmse_scorer,
                    "mae": mae_scorer,
                    "r2": r2_scorer,
                }
            score: dict[str, float] = cross_validate(
                regressor,
                X,
                y,
                cv=cv,
                scoring=scorers,
                # return_estimator=True,
                n_jobs=-1,
                # return_indices=True,
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
        regressor_params, X, y, cv,
        random_state:int,
        train_ratio:np.ndarray,
        scoring:str='r2',
    ) -> tuple:
     
    training_sizes, training_scores, testing_scores = learning_curve(
                                                        regressor_params,
                                                        X,
                                                        y,
                                                        cv=cv,
                                                        n_jobs=-1,
                                                        train_sizes=train_ratio,
                                                        scoring=scoring,
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

class _RF_Uncertainty:
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model

    def fit(self, X_train_u, y_train_u)->None:
        self.x_scaler= StandardScaler()
        self.y_scaler= StandardScaler()
        self.X_train_u = self.x_scaler.fit_transform(X_train_u)
        self.y_train_u = self.y_scaler.fit_transform(y_train_u)
        self.trained_u = self.fitted_model.fit(self.X_train_u, self.y_train_u)

    def pred_dist(self, X_test_u) -> np.ndarray:
        self.X_test_u = self.x_scaler.transform(X_test_u)
        all_preds = Parallel(n_jobs=-1)(
                delayed(tree.predict)(self.X_test_u) for tree in self.fitted_model.estimators_
                )
        return np.std(all_preds, axis=0)

    

class _XGB_Uncertainty:
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model
    def fit(self, X_train_u, y_train_u)->None:
        model_num = self.fitted_model.get_params().get('n_estimators', 500)
        self.x_scaler= StandardScaler()
        self.y_scaler= StandardScaler()
        self.X_train_u = self.x_scaler.fit_transform(X_train_u)
        self.y_train_u = self.y_scaler.fit_transform(y_train_u)
        self.boosted_reg = BaggingRegressor(estimator=self.fitted_model, n_estimators=model_num).fit(self.X_train_u,
                                                                                                    self.y_train_u)
    def pred_dist(self, X_test_u) -> np.ndarray:
        self.X_test_u = self.x_scaler.transform(X_test_u)
        all_preds = Parallel(n_jobs=-1)(
                delayed(tree.predict)(self.X_test_u ) for tree in self.boosted_reg.estimators_
                )
        return np.std(all_preds, axis=0)
    

def train_and_predict_ood(reg, X_train_val, y_train_val, X_test, y_test, 
                            algorithm:str, return_train_pred:bool=False,
                            manual_preprocessor:Pipeline=None) -> tuple:
    
    reg_u = reg.named_steps['regressor'].regressor
    reg.fit(X_train_val, y_train_val)
    if algorithm == 'NGB':
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_val_u = x_scaler.fit_transform(X_train_val)
        y_train_val_u = y_scaler.fit_transform(y_train_val)
        X_test_u = x_scaler.transform(X_test)
        reg_u.fit(X_train_val_u, y_train_val_u)
        y_test_predist = reg_u.pred_dist(X_test_u)
        y_test_pred_uncertainty = np.array(np.sqrt(y_test_predist.var)).reshape(y_test.shape)
        y_test_pred = reg.predict(X_test)

    elif algorithm == 'RF':
        uncertainty_estimator = _RF_Uncertainty(reg_u)
        uncertainty_estimator.fit(X_train_val, y_train_val)
        y_test_pred_uncertainty = uncertainty_estimator.pred_dist(X_test).reshape(y_test.shape)
        y_test_pred = reg.predict(X_test)
    elif algorithm == 'XGBR':
        xgb_u = _XGB_Uncertainty(reg_u)
        xgb_u.fit(X_train_val, y_train_val)
        y_test_pred_uncertainty = xgb_u.pred_dist(X_test).reshape(y_test.shape)
        y_test_pred = reg.predict(X_test)
    else:
        y_test_pred = reg.predict(X_test)
        y_test_pred_uncertainty = None

    
    test_scores = get_prediction_scores(y_test, y_test_pred,'test')
    if return_train_pred:
        y_train_pred = reg.predict(X_train_val)
        train_scores = get_prediction_scores(y_train_val, y_train_pred,'train')
        return test_scores, train_scores, y_test_pred, y_test_pred_uncertainty
    
    return test_scores, y_test_pred, y_test_pred_uncertainty


def get_prediction_scores(y_true, y_pred, score_set:str='test'):
    return {
        f"{score_set}_mad": np.abs(y_pred - y_pred.mean()).mean(),
        f"{score_set}_ystd": y_pred.std(),
        f"{score_set}_mae": mean_absolute_error(y_true, y_pred),
        f"{score_set}_rmse": root_mean_squared_error(y_true, y_pred),
        f"{score_set}_r2": r2_score(y_true, y_pred),
        # f"{score_set}_pearson_r": pearsonr(y_test, y_pred)[0],
        # f"{score_set}_pearson_p_value": pearsonr(y_test, y_pred)[1],
        # f"{score_set}_spearman_r": spearmanr(y_test, y_pred)[0],
        # f"{score_set}_spearman_p_value": spearmanr(y_test, y_pred)[1],
        # f"{score_set}_kendall_r": kendalltau(y_test, y_pred)[0],
        # f"{score_set}_kendall_p_value": kendalltau(y_test, y_pred)[1],
    }



