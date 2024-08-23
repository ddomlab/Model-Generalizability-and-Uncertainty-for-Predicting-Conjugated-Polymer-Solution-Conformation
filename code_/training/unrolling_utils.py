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
    print(f"this is n\ {rolled_cols.loc[1]}")
    unrolled_df: pd.DataFrame = pd.concat([rolled_cols[col].apply(pd.Series) for col in rolled_cols.columns], axis=1)
    unrolled_df.columns = new_col_names
    return unrolled_df


def unroll_ECFP(df: pd.DataFrame, col_names: list[str], oligomer_representation:str ,
                vector_type:str ,radius: int = 0, n_bits: int = 0,**kwargs) -> pd.DataFrame:
    
    new_ecfp_col_names: list[str] = [f"{oligomer_representation}_ECFP{2 * radius}_{vector_type}_bit{i}" for i in range(n_bits)]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_ecfp_col_names)
    return new_df





def unroll_Mordred_descriptors(df: pd.DataFrame, col_names: list[str], oligomer_representation:str,
                        **kwargs) -> pd.DataFrame:
    mordred_descriptors: pd.DataFrame = pd.DataFrame.from_records(df[col_names])
    mordred_descriptors: pd.DataFrame = mordred_descriptors.rename(columns=lambda x: f"{oligomer_representation}_Mordred{x}")
    return mordred_descriptors

def unroll_MACCS(df: pd.DataFrame, col_names: list[str], oligomer_representation:str,
                        **kwargs):
    new_ecfp_col_names: list[str] = [f"{oligomer_representation}_MACCS_bit{i}" for i in range(166)]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_ecfp_col_names)
    return new_df



radius_to_bits: dict[int, int] = {3: 512, 4: 1024, 5: 2048, 6: 4096}


unrolling_factory: dict[str, Callable] = {#"solvent":             unroll_solvent_descriptors,
                                          #"solvent additive":    unroll_solvent_descriptors,
                                          "ECFP":                 unroll_ECFP,
                                          "Mordred":              unroll_Mordred_descriptors,
                                          "MACCS":                unroll_MACCS,
                                          #"BRICS":               unroll_tokens,
                                          #"SELFIES":             unroll_tokens,
                                          #"SMILES":              unroll_tokens,
                                          #"OHE":                 get_ohe_structures,
                                          #"material properties": get_material_properties,
                                          #"graph embeddings":    get_gnn_embeddings,
                                          #"PUFp":                unroll_pufp,
                                          }










