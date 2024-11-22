from pathlib import Path
import pandas as pd
from clean_dataset import main_cleaning, drop_block_cp, map_derived_Rh_data, Rh_columns_to_map

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
RAW_dir = DATASETS/ 'raw'
JSONS =  DATASETS/'json_resources'

raw_dataset: pd.DataFrame = pd.read_excel(RAW_dir/'Polymer_Solution_Scattering_Dataset.xlsx') 
Rh_data: pd.DataFrame = pd.read_pickle(RAW_dir/'Rh distribution-intensity weighted.pkl')


if __name__ == "__main__":
    # print(Rh_data)
    # raw_dataset['intensity weighted average Rh (nm)'] = raw_dataset.index.to_series().apply(
    # map_intensity_weighted_rh, args=(Rh_data, raw_dataset,'intensity weighted average Rh (nm)'))
    
    # raw_dataset['intensity weighted average over log(Rh (nm))'] = raw_dataset.index.to_series().apply(
    # map_intensity_weighted_rh, args=(Rh_data, raw_dataset,'intensity weighted average over log(Rh (nm))'))

    mapped_dataset = map_derived_Rh_data(raw_dataset, Rh_data, Rh_columns_to_map)
    print("Done with mapping Rh data")
    cleaned_df = main_cleaning(mapped_dataset)
    print("Done with cleaning")
    drop_block_cp(cleaned_df)