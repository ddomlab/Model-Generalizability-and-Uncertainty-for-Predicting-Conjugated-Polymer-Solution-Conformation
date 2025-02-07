import json
from pathlib import Path
from types import NoneType
from typing import Dict, Optional,Tuple

import numpy as np
import pandas as pd
from all_factories import radius_to_bits

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent.parent

feature_abbrev: Dict[str, str] = {
    "Lp (nm)":          "Lp",
    "Rg1 (nm)":         "Rg",
    "Rh (IW avg log)":  "Rh",
    "Concentration (mg/ml)":            "concentration",
    "Temperature SANS/SLS/DLS/SEC (K)":         "temperature",
    "Mn (g/mol)":         "Mn", 
    "Mw (g/mol)":         "Mw", 
    "First Peak":         "Rh First Peak",
    "Second Peak":         "Rh Second Peak",
    "First Peak":         "Rh First Peak",
    "Third Peak":         "Rh Third Peak",
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


def _save(scores: Optional[Dict[int, Dict[str, float]]],
          predictions: Optional[pd.DataFrame],
          df_shapes: Optional[Dict],
          generalizability_score:Optional[Dict],
          results_dir: Path,
          regressor_type: str,
        #   hyperparameter_optimization: Optional[bool],
          imputer: Optional[str],
          representation: str,
          pu_type : Optional[str],
          radius : Optional[int],
          vector : Optional[str],
          numerical_feats: Optional[list[str]],
          hypop: bool=True,
          transform_type:Optional[str]=None
          ) -> None:
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # just scaler
    if numerical_feats and pu_type==None:
        short_num_feats = "-".join(feature_abbrev.get(key,key) for key in numerical_feats)
        fname_root = f"({short_num_feats})_{regressor_type}"
        fname_root = f"{fname_root}_{imputer}" if imputer else fname_root
    
    # scaler and structural mixed
    if numerical_feats and pu_type:
        short_num_feats = "-".join(feature_abbrev.get(key,key) for key in numerical_feats)
        if radius:
            fname_root = f"({representation}{radius}.{vector}.{radius_to_bits[radius]}-{short_num_feats})_{regressor_type}"
        
        else:
            fname_root = f"({representation}-{short_num_feats})_{regressor_type}"
            fname_root = f"{fname_root}_{imputer}" if imputer else fname_root

    #just structural
    if numerical_feats==None and pu_type: 
        if radius:
            fname_root = f"({representation}{radius}.{vector}.{radius_to_bits[radius]})_{regressor_type}"

        else:
            fname_root = f"({representation})_{regressor_type}"
    
    fname_root =f"{fname_root}_hypOFF" if hypop==False else fname_root
    fname_root =f"{fname_root}_{transform_type}" if transform_type else f"{fname_root}_transformerOFF"

    print("Filename:", fname_root)
    if scores:
        scores_file: Path = results_dir / f"{fname_root}_scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)
        print(scores_file)

    if predictions is not None and not predictions.empty:
        predictions_file: Path = results_dir / f"{fname_root}_predictions.csv"
        predictions.to_csv(predictions_file, index=False)
        print(predictions_file)

    
    if df_shapes:
        data_shape_file:Path = results_dir / f"{fname_root}_shape.json"
        with open(data_shape_file, "w") as f:
            json.dump(df_shapes, f, cls=NumpyArrayEncoder, indent=2)
        print(data_shape_file)
    
    if generalizability_score:
        generalizibility_scores_file: Path = results_dir / f"{fname_root}_generalizability_scores.json"
        with open(generalizibility_scores_file, "w") as f:
            json.dump(generalizability_score, f, cls=NumpyArrayEncoder, indent=2)
        print(generalizibility_scores_file)
    
    print('Done Saving scores!')


def save_results(scores:Optional[Dict[int, Dict[str, float]]]=None,
                 predictions: Optional[pd.DataFrame]=None,
                 df_shapes:Optional[Dict]=None,
                 generalizability_score:Optional[Dict]=None,
                 target_features: list=None,
                 regressor_type: str=None,
                 kernel: Optional[str]=None,
                 TEST : bool =True,
                 representation: str=None,
                 pu_type : Optional[str]=None,
                 radius : Optional[int]=None,
                 vector : Optional[str]=None,
                 numerical_feats: Optional[list[str]]=None,
                 imputer: Optional[str] = None,
                 output_dir_name: str = "results",
                 cutoff: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] =None,
                 hypop: Optional[bool]=True,
                 transform_type:Optional[str]=None,
                 second_transformer:Optional[str]=None,
                 classification:bool=None
                 ) -> None:
    
    targets_dir: str = "-".join([feature_abbrev.get(target, target) for target in target_features])
    feature_ids = []
    
    regressor_type = f"{regressor_type}.{kernel}" if kernel is not None else regressor_type
    
    if pu_type:
        feature_ids.append(pu_type)
    if numerical_feats:
        feature_ids.append('scaler')
    features_dir: str = "_".join(feature_ids)
    print(features_dir)
    if cutoff:
        cutoff_parameter = "-".join(feature_abbrev.get(key,key) for key in cutoff)
    f_root_dir = f"classification_target_{targets_dir}" if classification else  f"target_{targets_dir}"
    f_root_dir = f"{f_root_dir}_{second_transformer}FT" if second_transformer else f_root_dir
    f_root_dir = f"{f_root_dir}_filter_({cutoff_parameter})" if cutoff else f_root_dir

    results_dir: Path = ROOT / output_dir_name / f_root_dir
    results_dir: Path = results_dir / "test" if TEST else results_dir
    results_dir: Path = results_dir / features_dir
    # if subspace_filter:
    #     results_dir = results_dir / f"subspace_{subspace_filter}"

    _save(scores=scores,
          predictions=predictions,
          results_dir=results_dir,
          df_shapes=df_shapes,
          generalizability_score=generalizability_score,
          regressor_type=regressor_type,
          imputer=imputer,
          representation=representation,
          pu_type=pu_type,
          radius=radius,
          vector=vector,
          numerical_feats=numerical_feats,
          hypop=hypop,
          transform_type=transform_type
          )


