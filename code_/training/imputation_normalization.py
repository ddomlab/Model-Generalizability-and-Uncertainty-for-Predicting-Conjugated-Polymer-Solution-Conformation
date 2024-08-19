def preprocessing_workflow(imputer: Optional[str],
                           feat_to_impute: Optional[list[str]] = None,
                           representation: Optional[str] = None,
                           numerical_feat: Optional[list] = None,
                           structural_feat: Optional[list] = None,
                           special_column: Optional[str] = None,
                           scaler: str = "Standard",
                           ) -> Pipeline:
    # structure_feats and scalar_feats for scaling
    # all_columns = list(set(columns_to_impute)|set(special_column))
    # imputation of columns
    steps = []
    if imputer:
        steps = [
            ("Impute feats", ColumnTransformer(
                transformers=[
                    (f'imputer_{imputer}', imputer_factory[imputer], feat_to_impute),
                ],
                remainder="passthrough", verbose_feature_names_out=False
            )),
            # put if special column:
            ("Calculate Mw", ColumnTransformer(
                transformers=[
                    (f'calculator_{special_column}',
                     (FunctionTransformer(calculate_mw,
                                          kw_args={'mw': special_column, 'mn': 'Mn (g/mol)', 'pdi': 'PDI'},
                                          validate=False)), ['Mn (g/mol)', special_column, 'PDI'])
                ],
                remainder="passthrough", verbose_feature_names_out=False
            )),

            ("Impute the rest of Mw values", ColumnTransformer(
                transformers=[
                    (f"imputer_{special_column}", imputer_factory[imputer], [special_column])
                ],
                remainder="passthrough", verbose_feature_names_out=False
            )),
            ("drop Mn", ColumnTransformer(
                transformers=[
                    ("Drop Mn column",
                     FunctionTransformer(drop_columns, kw_args={'columns_to_drop': ['Mn (g/mol)']}, validate=False),
                     ['Mn (g/mol)'])
                ], remainder="passthrough", verbose_feature_names_out=False))
        ]
        # scaler
    transformers = []
    if numerical_feat:
        transformers.append(
            ("numerical_scaling", transforms[scaler], numerical_feat)
        )
    elif representation_scaling_factory[representation]['callable']:
        transformers.append(
            ("structural_scaling", representation_scaling_factory[representation]['callable'], structural_feat)
        )

    scaler = ("scaling features",
              ColumnTransformer(transformers=[*transformers], remainder="passthrough", verbose_feature_names_out=False)
              )
    steps.append(scaler)
    return Pipeline(steps)