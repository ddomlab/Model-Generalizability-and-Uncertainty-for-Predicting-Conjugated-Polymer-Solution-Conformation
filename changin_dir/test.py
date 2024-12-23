import os
from pathlib import Path
import re
import pandas as pd

HERE: Path = Path(__file__).resolve().parent
datasets: Path = HERE.parent/ 'datasets'
training_df_dir: Path = datasets/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped.pkl"

w_data = pd.read_pickle(training_df_dir)

zero_count = (w_data['Rh (IW avg log)'] <= 0).sum()
print(f"Number of rows with zero in 'Rh (IW avg log)': {zero_count}")