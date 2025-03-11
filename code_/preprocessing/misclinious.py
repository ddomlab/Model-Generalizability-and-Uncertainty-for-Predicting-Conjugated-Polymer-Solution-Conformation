import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from visualization.visualization_setting import save_img_path


HERE: Path = Path(__file__).resolve().parent
VISUALIZATION = HERE.parent/ "visualization"
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "Rg data with clusters.pkl"
w_data = pd.read_pickle(training_df_dir)
w_data = w_data.reset_index(drop=True)
w_data['log Rg (nm)'] = np.log10(w_data['Rg1 (nm)'])

w_data.to_pickle(training_df_dir)

