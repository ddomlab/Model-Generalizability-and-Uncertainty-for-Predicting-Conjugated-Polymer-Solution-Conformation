import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import DBSCAN
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from visualization.visualization_setting import (set_plot_style,
                                                 save_img_path)
set_plot_style(tick_size=18)

HERE: Path = Path(__file__).resolve().parent
VISUALIZATION = HERE.parent.parent/ "visualization"
DATASETS: Path = HERE.parent.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal (40-1000 nm)_added.pkl"
w_data = pd.read_pickle(training_df_dir)