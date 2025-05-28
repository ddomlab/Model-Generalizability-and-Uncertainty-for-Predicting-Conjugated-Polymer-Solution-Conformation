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
    "canonical_name": "Polymers cluster",
    "Dark/light": "light exposure",
    "Aging time (hour)": "aging time",
    "To Aging Temperature (K)": "aging temperature",
    "Sonication/Stirring/heating Temperature (K)": "Prep temperature",
    "Merged Stirring /sonication/heating time(min)": "Prep time",
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

def get_cv_splits(score_for_indices):
    indices = {}
    for seed, values in score_for_indices.items():
        if isinstance(seed, int) and "indices" in values:
            indices[seed] = values["indices"]
    return indices

def _save(scores: Optional[Dict[int, Dict[str, float]]],
          predictions: Optional[pd.DataFrame],
          ground_truth: Optional[Dict],
          df_shapes: Optional[Dict],
          results_dir: Path,
          regressor_type: str,
          imputer: Optional[str],
          representation: str,
          pu_type : Optional[str],
          radius : Optional[int],
          vector : Optional[str],
          numerical_feats: Optional[list[str]],
          hypop: bool=True,
          transform_type:Optional[str]=None,
          learning_curve:bool=False,
          special_file_name:Optional[str]=None,
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
    fname_root = f"{fname_root}_lc" if learning_curve else fname_root
    fname_root = f"{fname_root}_{special_file_name}" if special_file_name else fname_root
    print("Filename:", fname_root)

    if scores:
        # cv_indices = get_cv_splits(scores)
        # print("CV Indices:", cv_indices)
        scores_file: Path = results_dir / f"{fname_root}_scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)
        
        # indices_file: Path = results_dir / f"{fname_root}_indices.json"
        # with open(indices_file, "w") as f:
        #     json.dump(cv_indices, f, cls=NumpyArrayEncoder, indent=2)

    if predictions is not None:
        if isinstance(predictions, pd.DataFrame):
            predictions_file: Path = results_dir / f"{fname_root}_predictions.csv"
            predictions.to_csv(predictions_file, index=False)
        elif isinstance(predictions, dict):
            predictions_file: Path = results_dir / f"{fname_root}_predictions.json"
            
            with open(predictions_file, "w") as f:
                json.dump(predictions, f, cls=NumpyArrayEncoder, indent=2)


    if ground_truth:
        # print(ground_truth)
        cluster_ground_truth:Path = results_dir / f"{fname_root}_ClusterTruth.json"
        with open(cluster_ground_truth, "w") as f:
            json.dump(ground_truth, f, cls=NumpyArrayEncoder, indent=2)
    
    if df_shapes:
        data_shape_file:Path = results_dir / f"{fname_root}_shape.json"
        with open(data_shape_file, "w") as f:
            json.dump(df_shapes, f, cls=NumpyArrayEncoder, indent=2)

    
    print('Done Saving scores!')


def save_results(scores:Optional[Dict[int, Dict[str, float]]]=None,
                 predictions: Optional[pd.DataFrame]=None,
                 ground_truth: Optional[pd.DataFrame]=None,
                 df_shapes:Optional[Dict]=None,
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
                 classification:bool=False,
                 clustering_method:str=None,
                 learning_curve:bool=False,
                 special_folder_name:Optional[str]=None,
                 special_file_name:Optional[str]=None,
                 ) -> None:
    
    targets_dir: str = "-".join([feature_abbrev.get(target, target) for target in target_features])
    feature_ids = []
    
    regressor_type = f"{regressor_type}.{kernel}" if kernel is not None else regressor_type
    
    if pu_type:
        feature_ids.append(pu_type)
    if numerical_feats:
        feature_ids.append('scaler')
    features_dir: str = "_".join(feature_ids)
    if cutoff:
        cutoff_parameter = "-".join(feature_abbrev.get(key,key) for key in cutoff)
    f_root_dir = f"classification_target_{targets_dir}" if classification else  f"target_{targets_dir}"
    f_root_dir = f"OOD_{f_root_dir}" if clustering_method else f_root_dir
    f_root_dir = f"{f_root_dir}_{second_transformer}FT" if second_transformer else f_root_dir
    f_root_dir = f"{f_root_dir}_filter_({cutoff_parameter})" if cutoff else f_root_dir
    f_root_dir = f"{f_root_dir}_{special_folder_name}" if special_folder_name else f_root_dir

    results_dir: Path = ROOT / output_dir_name / f_root_dir
    clustering_method= feature_abbrev.get(clustering_method, clustering_method) if clustering_method else None
    results_dir: Path = results_dir / clustering_method if clustering_method else results_dir
    results_dir: Path = results_dir / "test" if TEST else results_dir
    results_dir: Path = results_dir / features_dir
    # if subspace_filter:
    #     results_dir = results_dir / f"subspace_{subspace_filter}"

    _save(scores=scores,
          predictions=predictions,
          ground_truth=ground_truth,
          results_dir=results_dir,
          df_shapes=df_shapes,
          regressor_type=regressor_type,
          imputer=imputer,
          representation=representation,
          pu_type=pu_type,
          radius=radius,
          vector=vector,
          numerical_feats=numerical_feats,
          hypop=hypop,
          transform_type=transform_type,
          learning_curve=learning_curve,
          special_file_name=special_file_name,
          )
    return results_dir

