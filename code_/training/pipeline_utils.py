import json
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, QuantileTransformer, StandardScaler

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"

scaler_factory: dict[str, type] = {"MinMax": MinMaxScaler, "Standard": StandardScaler}


def unroll_lists_to_columns(df: pd.DataFrame, unroll_cols: list[str], new_col_names: list[str]) -> pd.DataFrame:
    """
    Unroll a list of lists into columns of a DataFrame.

    Args:
        df: DataFrame to unroll.
        unroll_cols: List of columns containing list to unroll.
        new_col_names: List of new column names.

    Returns:
        DataFrame with unrolled columns.
    """
    rolled_cols: pd.DataFrame = df[unroll_cols]
    # rolled_cols: pd.DataFrame = df
    unrolled_df: pd.DataFrame = pd.concat([rolled_cols[col].apply(pd.Series) for col in rolled_cols.columns], axis=1)
    unrolled_df.columns = new_col_names
    return unrolled_df


def unroll_fingerprints(df: pd.DataFrame, col_names: list[str] = [], radius: int = 0, n_bits: int = 0,
                        **kwargs) -> pd.DataFrame:
    new_ecfp_col_names: list[str] = [f"EFCP{2 * radius}_bit{i}" for i in range(n_bits)]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_ecfp_col_names)
    return new_df


def unroll_solvent_descriptors(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    with open(DATASETS / "Min_2020_n558" / "selected_properties.json", "r") as f:
        solvent_descriptors: list[str] = json.load(f)["solvent"]

    solvent_cols: list[str] = ["solvent descriptors", "solvent additive descriptors"]
    solvent_descriptor_cols: list[str] = [*[f"solvent {d}" for d in solvent_descriptors],
                                         *[f"solvent additive {d}" for d in solvent_descriptors]
                                         ]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, solvent_cols, solvent_descriptor_cols)
    return new_df


# def unroll_pufp(df: pd.DataFrame, col_names: list[str] = [], **kwargs) -> pd.DataFrame:
#     new_pufp_col_names: list[str] = [*[f"D PUFp_bit{i}" for i in range(147)],  # 148 is the number of bits in the donor PUFp fingerprint
#                                      *[f"A PUFp_bit{i}" for i in range(195)]]  # 145 is the number of bits in the PUFp fingerprint
#     new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_pufp_col_names)
#     return new_df


# def get_token_len(df: pd.DataFrame) -> list[int]:
#     token_lens: list[int] = []
#     for col in df.columns:
#         tokenized = df.loc[0, col]
#         pass
#         if isinstance(tokenized, list):
#             n_tokens: int = len(tokenized)
#         else:
#             n_tokens: int = 1
#         token_lens.append(n_tokens)
#     return token_lens


# def unroll_tokens(df: pd.DataFrame, col_names: list[str] = [], **kwargs) -> pd.DataFrame:
#     num_tokens: list[int] = get_token_len(df)
#     token_type: str = df.columns[0].split(" ")[1]
#     new_token_col_names: list[str] = [*[f"D {token_type} {i}" for i in range(num_tokens[0])],
#                                       *[f"A {token_type} {i}" for i in range(num_tokens[1])]]
#     new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_token_col_names)
#     return new_df


# def get_ohe_structures(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#     # df = df[["Donor", "Acceptor"]]
#     ohe: OneHotEncoder = OneHotEncoder(sparse=False).set_output(transform="pandas")
#     new_df: pd.DataFrame = ohe.fit_transform(df)
#     return new_df


def get_mordred_descriptors(*args, data_file: Optional[Path] = None, **kwargs) -> pd.DataFrame:
    default_mordred: Path = DATASETS / "Min_2020_n558" / "cleaned_dataset_mordred.pkl"
    mordred_file: Path = data_file if data_file is not None else default_mordred
    # with open(mordred_file, "rb") as f:
    mordred: pd.DataFrame = pd.read_pickle(mordred_file)
    return mordred


# def get_gnn_embeddings(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#     with open(DATASETS / "Min_2020_n558" / "cleaned_dataset_gnnembeddings.pkl", "rb") as f:
#         graph_embeddings: pd.DataFrame = pd.read_pickle(f)
#     return graph_embeddings


# def get_material_properties(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#     return df



radius_to_bits: dict[int, int] = {3: 512, 4: 1024, 5: 2048, 6: 4096}

# power_features: list[str] = [
#     "Donor PDI", "Donor Mn (kDa)", "Donor Mw (kDa)",
#     "HOMO_D (eV)", "LUMO_D (eV)", "Eg_D (eV)", "Ehl_D (eV)",
#     "HOMO_A (eV)", "LUMO_A (eV)", "Eg_A (eV)", "Ehl_A (eV)",
#     "active layer thickness (nm)",
#     "log hole mobility blend (cm^2 V^-1 s^-1)", "log electron mobility blend (cm^2 V^-1 s^-1)",
#     "log hole:electron mobility ratio",
# ]


transforms: dict[str, Callable] = {
    None:                None,
    "MinMax":            MinMaxScaler,
    "Standard":          StandardScaler,
    "Power":             PowerTransformer,
    "Uniform Quantile":  QuantileTransformer,
}


imputer_factory: dict[str, _BaseImputer] = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "most-frequent": SimpleImputer(strategy="most_frequent"),
    "uniform KNN": KNNImputer(weights="uniform"),
    "distance KNN": KNNImputer(weights="distance"),
    "iterative": IterativeImputer(sample_posterior=True),
}

unrolling_factory: dict[str, Callable] = {#"solvent":             unroll_solvent_descriptors,
                                          #"solvent additive":    unroll_solvent_descriptors,
                                          "ECFP":                 unroll_fingerprints,
                                          #"mordred":             get_mordred_descriptors,
                                          #"BRICS":               unroll_tokens,
                                          #"SELFIES":             unroll_tokens,
                                          #"SMILES":              unroll_tokens,
                                          #"OHE":                 get_ohe_structures,
                                          #"material properties": get_material_properties,
                                          #"graph embeddings":    get_gnn_embeddings,
                                          #"PUFp":                unroll_pufp,
                                          }


def calculate_mw(df: pd.DataFrame,  # df containing Mw, Mn & PDI columns only
                 mw: str  , mn: str  , pdi: str   # Column names for Mw, Mn & PDI
                 ) -> pd.DataFrame:

  df.loc[df[mw].isna(), mw] = df[pdi] * df[mn]  # df.loc may give you some issues
  return df



def preprocessing_workflow(imputer:Optional[str],
                            feat_to_impute:Optional[list[str]]=None,
                            representation:Optional[str]=None,
                            numerical_feat:Optional[list]=None,
                            structural_feat:Optional[list]=None,
                            special_column:Optional[str] = None,
                            scaler: str= "Standard",
                            ) -> Pipeline:

      # structure_feats and scalar_feats for scaling
      # all_columns = list(set(columns_to_impute)|set(special_column))
      # imputation of columns
      steps = []
      if imputer:

          steps=[
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
                    (FunctionTransformer(calculate_mw, kw_args={'mw':special_column , 'mn': 'Mn (g/mol)', 'pdi': 'PDI'}, validate=False)),['Mn (g/mol)',special_column,'PDI'])
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
                  ("Drop Mn column", FunctionTransformer(drop_columns, kw_args={'columns_to_drop': ['Mn (g/mol)']}, validate=False), ['Mn (g/mol)'])
              ], remainder="passthrough", verbose_feature_names_out=False))
          ]
          # scaler
      transformers =[]
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

