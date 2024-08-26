import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from unrolling_utils import unrolling_factory

# HERE: Path = Path(__file__).resolve().parent
# DATASETS: Path = HERE.parent.parent / "datasets"

# with open(HERE / "filters.json", "r") as f:
#     FILTERS: dict[str, list[str]] = json.load(f)

# with open(HERE / "subsets.json", "r") as f:
#     SUBSETS: dict[str, list[str]] = json.load(f)





def sanitize_dataset(
    training_features: pd.DataFrame, targets: pd.DataFrame, dropna: bool, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanitize the training features and targets in case the target features contain NaN values.

    Args:
        training_features: Training features.
        targets: Targets.
        dropna: Whether to drop NaN values.
        **kwargs: Keyword arguments to pass to filter_dataset.

    Returns:
        Sanitized training features and targets.
    """
    if dropna:
        targets: pd.DataFrame = targets.dropna()
        training_features: pd.DataFrame =training_features.loc[targets.index]
        return training_features, targets
    else:
        return training_features, targets


def filter_dataset(
    raw_dataset: pd.DataFrame,
    structure_feats: Optional[list[str]], # can be None
    scalar_feats: Optional[list[str]], # like conc, temp
    target_feats: list[str], # lp
    dropna: bool = True,
    unroll: Union[dict, list, None] = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Filter the dataset.

    Args:
        raw_dataset: Raw dataset.
        structure_feats: Structure features.
        scalar_feats: Scalar features.
        target_feats: Target features.

    Returns:
        Input features and targets.
    """
    # Add multiple lists together as long as they are not NoneType
    all_feats: list[str] = [
        feat
        for feat_list in [structure_feats, scalar_feats, target_feats]
        if feat_list
        for feat in feat_list
    ]

    dataset: pd.DataFrame = raw_dataset[all_feats]



    if unroll:
        if isinstance(unroll, dict):
            structure_features: pd.DataFrame = unrolling_factory[
                unroll["representation"]](dataset[structure_feats], **unroll)
        elif isinstance(unroll, list):
            multiple_unrolled_structure_feats: list[pd.DataFrame] = []
            for unroll_dict in unroll:
                single_structure_feat: pd.DataFrame = filter_dataset(
                    dataset,
                    # structure_feats=unroll_dict["columns"],
                    structure_feats=unroll_dict["col_names"],
                    scalar_feats=[],
                    target_feats=[],
                    # dropna=dropna,
                    dropna=False,
                    unroll=unroll_dict,
                )[0]
                multiple_unrolled_structure_feats.append(single_structure_feat)
            structure_features: pd.DataFrame = pd.concat(
                multiple_unrolled_structure_feats, axis=1
            )
        else:
            raise ValueError(f"Unroll must be a dict or list, not {type(unroll)}")
    elif structure_feats:
        structure_features: pd.DataFrame = dataset[structure_feats]
    else:
        structure_features: pd.DataFrame = dataset[[]]

    if scalar_feats:
      scalar_features: pd.DataFrame = dataset[scalar_feats]
    else:
      scalar_features: pd.DataFrame = dataset[[]]

    training_features: pd.DataFrame = pd.concat(
        [structure_features, scalar_features], axis=1
    )

    targets: pd.DataFrame = dataset[target_feats]

    training_features, targets = sanitize_dataset(
        training_features, targets, dropna=dropna, **kwargs
    )

    # if not (scalars_available and struct_available):
    new_struct_feats: list[str] = structure_features.columns.tolist()
    return training_features, targets, new_struct_feats
