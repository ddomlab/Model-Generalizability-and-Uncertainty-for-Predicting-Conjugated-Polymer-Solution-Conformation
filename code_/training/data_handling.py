import json
from pathlib import Path
from types import NoneType
from typing import Dict, Optional,Tuple

import numpy as np
import pandas as pd
from all_factories import radius_to_bits

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent.parent

target_abbrev: Dict[str, str] = {
    "Lp (nm)":          "Lp",
    "Rg1 (nm)":         "Rg",
    "Voc (V)":            "Voc",
    "Jsc (mA cm^-2)":     "Jsc",
    "FF (%)":             "FF",
    "Concentration (mg/ml)": "Concentration"
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
    
    if numerical_feats and pu_type==None:
        fname_root = f"(numerical)_{regressor_type}"
        fname_root = f"{fname_root}_{imputer}" if imputer else fname_root
    if numerical_feats and pu_type:
        if radius:
            fname_root = f"({representation}{radius})_{vector}_{radius_to_bits[radius]}_numerical)_{regressor_type}"
        
        else:
            fname_root = f"({representation}_numerical)_{regressor_type}"
            fname_root = f"{fname_root}_{imputer}" if imputer else fname_root

    if numerical_feats==None and pu_type: 
        if radius:
            fname_root = f"({representation}{radius})_{vector}_{radius_to_bits[radius]}_{regressor_type}"

        else:
            fname_root = f"({representation})_{regressor_type}"
    

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
                 target_features: list,
                 regressor_type: str,
                 TEST : bool =True,
                 representation: str=None,
                 pu_type : Optional[str]=None,
                 radius : Optional[int]=None,
                 vector : Optional[str]=None,
                 numerical_feats: Optional[list[str]]=None,
                 imputer: Optional[str] = None,
                 output_dir_name: str = "results",
                 cutoff: Dict[str, Tuple[Optional[float], Optional[float]]] =None,
                 ) -> None:
    
    targets_dir: str = "-".join([target_abbrev[target] for target in target_features])
    feature_ids = []
    if pu_type:
        feature_ids.append(pu_type)
    if numerical_feats:
        feature_ids.append('scaler')
    features_dir: str = "_".join(feature_ids)
    print(features_dir)
    if cutoff:
        cutoff_parameter = "-".join( target_abbrev[key] for key in cutoff)
    
    f_root_dir = f"target_{targets_dir}"
    f_root_dir = f"{f_root_dir}_filter_({cutoff_parameter})" if cutoff else f_root_dir

    results_dir: Path = ROOT / output_dir_name / f_root_dir
    results_dir: Path = results_dir / "test" if TEST else results_dir
    results_dir: Path = results_dir / features_dir
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


