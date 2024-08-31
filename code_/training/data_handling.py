import json
from pathlib import Path
from types import NoneType
from typing import Dict, Optional

import numpy as np
import pandas as pd
from all_factories import radius_to_bits

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent.parent

target_abbrev: Dict[str, str] = {
    "Lp (nm)":          "Lp",
    "Voc (V)":            "Voc",
    "Jsc (mA cm^-2)":     "Jsc",
    "FF (%)":             "FF",
}


def remove_unserializable_keys(d: Dict) -> Dict:
    """Remove unserializable keys from a dictionary."""
    # for k, v in d.items():
    #     if not isinstance(v, (str, int, float, bool, NoneType, tuple, list, np.ndarray, np.floating, np.integer)):
    #         d.pop(k)
    #     elif isinstance(v, dict):
    #         d[k] = remove_unserializable_keys(v)
    # return d
    new_d: dict = {k: v for k, v in d.items() if
                   isinstance(v, (str, int, float, bool, NoneType, np.floating, np.integer))}
    return new_d


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def _save(scores: Dict[int, Dict[str, float]],
          predictions: Optional[pd.DataFrame],
          results_dir: Path,
          regressor_type: str,
        #   hyperparameter_optimization: Optional[bool],
          imputer: Optional[str],
          representation: str,
          pu_type : Optional[str],
          radius : Optional[int],
          vector : Optional[str],
          numerical_feats: Optional[list[str]]
          ) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    fname_root = f"{regressor_type} model_{imputer} imputer" if imputer else regressor_type
    fname_root = f"{fname_root}_numerical_feats" if numerical_feats else fname_root
    if representation:
        fname_root = f"{fname_root}_{representation}" if representation else fname_root
        fname_root = f"{fname_root}{radius}_{vector}_{radius_to_bits[radius]}bits" if representation=='ECFP' else fname_root
        fname_root = f"{fname_root}_{pu_type}" if pu_type else fname_root


    print("Filename:", fname_root)

    scores_file: Path = results_dir / f"{fname_root}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)

    predictions_file: Path = results_dir / f"{fname_root}_predictions.csv"
    predictions.to_csv(predictions_file, index=False)
    print("Saved results to:")
    print(scores_file)
    print(predictions_file)


def save_results(scores: dict,
                 predictions: pd.DataFrame,
                 target_features: str,
                 regressor_type: str,
                 TEST : bool =True,
                 representation: str=None,
                 pu_type : Optional[str]=None,
                 radius : Optional[int]=None,
                 vector : Optional[str]=None,
                 numerical_feats: Optional[list[str]]=None,
                 imputer: Optional[str] = None,
                 output_dir_name: str = "results",
                 ) -> None:
    targets_dir: str = "-".join([target_abbrev[target] for target in target_features])
    feature_ids = []
    if representation:
        feature_ids.append(representation)
    if numerical_feats:
        feature_ids.append('numerical')
    features_dir: str = "-".join(feature_ids)
    print(features_dir)
    results_dir: Path = ROOT / output_dir_name / f"target_{targets_dir}"
    results_dir: Path = results_dir / "test" if TEST else results_dir
    results_dir: Path = results_dir / f"{regressor_type} model" / features_dir
    # if subspace_filter:
    #     results_dir = results_dir / f"subspace_{subspace_filter}"

    _save(scores, predictions,
          results_dir=results_dir,
          regressor_type=regressor_type,
          imputer=imputer,
          representation= representation,
          pu_type=pu_type,
          radius=radius ,
          vector=vector ,
          numerical_feats=numerical_feats 
          )


