from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from ngboost import NGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from GPR_model import GPRegressor
# from sklearn.multioutput import MultiOutputRegressor

from sklearn.preprocessing import (StandardScaler,
                                   QuantileTransformer,
                                   MinMaxScaler,
                                   FunctionTransformer,
                                   RobustScaler)
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from typing import Callable, Optional, Union, Dict
from types import NoneType
from skopt.space import Integer, Real, Categorical



cutoffs = {
            # "Rh1 (nm)":1000,
            # "Rg1 (nm)":1000,
            # "Lc (nm)":1000,
            # "Lp (nm)":(None,100),
            # "Concentration (mg/ml)":(None,50)
        }



imputer_factory: Dict[str, TransformerMixin] = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "most-frequent": SimpleImputer(strategy="most_frequent"),
    "uniform KNN": KNNImputer(weights="uniform"),
    "distance KNN": KNNImputer(weights="distance"),
    "iterative": IterativeImputer(sample_posterior=True),
}

transforms: dict[str, Callable] = {
    None:                None,
    "MinMax":               MinMaxScaler(),
    "Standard":             StandardScaler(),
    "Robust Scaler":        RobustScaler(),
    "Uniform Quantile":     QuantileTransformer(),
}


radius_to_bits: dict[int, int] = {3: 512, 4: 1024, 5: 2048, 6: 4096}



representation_scaling_factory: dict[str, dict[str, Union[Callable, str]]] = {
    "ECFP":                {"callable": None, "type": None},
    "MACCS":                {"callable": None, "type": None},
    "Mordred":             {"callable": StandardScaler(),
                            "type":     "Standard"},
    # "graph embeddings":    {"callable": MinMaxScaler,
    #                         "type": "MinMax"},
    # "BRICS":               {"callable": MinMaxScaler, "type": "MinMax"},
    # "SELFIES":             {"callable": MinMaxScaler, "type": "MinMax"},
    # "SMILES":              {"callable": MinMaxScaler, "type": "MinMax"},
    # "OHE":                 {"callable": MinMaxScaler, "type": "MinMax"},
    # "GNN":     {"callable": MinMaxScaler, "type": "MinMax"},
}


regressor_factory: dict[str, type]={
    "MLR": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "XGBR": XGBRegressor(),
    "RF": RandomForestRegressor(),
    "Lasso": Lasso(),
    "DT": DecisionTreeRegressor(),
    "NGB": NGBRegressor(),
    "GPR": GPRegressor,
}







regressor_search_space = {
    "MLR": {
        "regressor__regressor__fit_intercept": [True, False]
    },
    "Lasso": {
        "regressor__alpha": Real(1e-3, 1e3, prior="log-uniform"),
        "regressor__fit_intercept": [True, False],
        "regressor__selection": Categorical(["cyclic", "random"]),
    },
    "KNN": {
        "regressor__n_neighbors": Integer(1, 50),
        "regressor__weights": Categorical(["uniform", "distance"]),
        "regressor__algorithm": Categorical(["ball_tree", "kd_tree", "brute"]),
        "regressor__leaf_size": Integer(1, 100),
    },
    "SVR": {
        "regressor__kernel": Categorical(["linear", "rbf"]),
    },
    "RF": {
        "regressor__regressor__n_estimators": Integer(50, 2000, prior="log-uniform"),
        "regressor__regressor__max_depth": [None],
        "regressor__regressor__min_samples_split": Real(0.05, 0.99),
        "regressor__regressor__min_samples_leaf": Real(0.05, 0.99),
        "regressor__regressor__max_features": Categorical(["sqrt", "log2"]),
    },
    "XGBR": {
        "regressor__regressor__n_estimators": Integer(50, 2000, prior="log-uniform"),
        "regressor__regressor__max_depth": Integer(10, 10000, prior="log-uniform"),
        # "regressor__grow_policy": Categorical(["depthwise", "lossguide"]),
        "regressor__regressor__n_jobs": [-2],
        "regressor__regressor__learning_rate": Real(1e-3, 1e-1, prior="log-uniform"),
    },
    "DT": {
        "regressor__regressor__min_samples_split": Real(0.05, 0.99),
        "regressor__regressor__min_samples_leaf": Real(0.05, 0.99),
        "regressor__regressor__max_features": Categorical([None,"sqrt", "log2"]),
        "regressor__regressor__max_depth": [None],
        "regressor__regressor__ccp_alpha": Real(0.05, 0.99),
    },
    "NGB": {
        "regressor__regressor__n_estimators": Integer(50, 2000, prior="log-uniform"),
        "regressor__regressor__learning_rate": Real(1e-6, 1e-3, prior="log-uniform"),
        "regressor__regressor__minibatch_frac": [1],
        # "regressor__regressor__minibatch_size":   Integer(1, 100),
        #  "regressor__regressor__Base":             Categorical(["DecisionTreeRegressor", "Ridge", "Lasso",
        #                                            "KernelRidge", "SVR"]),
        "regressor__regressor__natural_gradient": [True],
        "regressor__regressor__verbose": [False],
        # "regressor__regressor__tol": Real(1e-6, 1e-3, prior="log-uniform"),
    },
    "GPR": {
        "regressor__regressor__lr": [1e-2], 
        "regressor__regressor__n_epoch": [1000],

    }
}



results = {
    "model": [],
    "best_params":[],
    "imputer": [],
    "cv_score":[],
    "r2_train":[],
    "rmse_train":[],
    "r2_test":[],
    "rmse_test":[]
}
