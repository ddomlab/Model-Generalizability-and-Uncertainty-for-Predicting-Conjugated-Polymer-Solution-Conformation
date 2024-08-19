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
    "MinMax":            MinMaxScaler(),
    "Standard":          StandardScaler(),
    "Power":             PowerTransformer(),
    "Uniform Quantile":  QuantileTransformer(),
}





representation_scaling_factory: dict[str, dict[str, Union[Callable, str]]] = {
    "solvent":             {"callable": StandardScaler,
                            "type":     "Standard"},
    "ECFP":                {"callable": None, "type": "MinMax"},
    "PUFp":                {"callable": MinMaxScaler, "type": "MinMax"},
    "mordred":             {"callable": StandardScaler,
                            "type":     "Standard"},
    "graph embeddings":    {"callable": MinMaxScaler,
                            "type": "MinMax"},
    "BRICS":               {"callable": MinMaxScaler, "type": "MinMax"},
    "SELFIES":             {"callable": MinMaxScaler, "type": "MinMax"},
    "SMILES":              {"callable": MinMaxScaler, "type": "MinMax"},
    "OHE":                 {"callable": MinMaxScaler, "type": "MinMax"},
    "material properties": {"callable": StandardScaler, "type": "Standard"},
    "fabrication only":    {"callable": StandardScaler,
                            "type":     "Standard"},
    "GNN":     {"callable": MinMaxScaler, "type": "MinMax"},
}


regressor_factory: dict[str, type]={
    "MLR": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "XGBR": XGBRegressor(),
    "RF": RandomForestRegressor(),
    "Lasso": Lasso()
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
