from pathlib import Path
import pandas as pd
from clean_dataset import main_cleaning, drop_block_cp, assign_intensity_weighted_Rh

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
RAW_dir = DATASETS/ 'raw'
JSONS =  DATASETS/'json_resources'

raw_dataset: pd.DataFrame = pd.read_excel(RAW_dir/'Polymer_Solution_Scattering_Dataset.xlsx') 
Rh_data: pd.DataFrame = pd.read_csv(RAW_dir/'Rh distribution-intensity weighted.csv')


if __name__ == "__main__":
    cleaned_df = main_cleaning(raw_dataset)
    cleaned_df = assign_intensity_weighted_Rh(cleaned_df,rh_data=Rh_data)
    drop_block_cp(cleaned_df)