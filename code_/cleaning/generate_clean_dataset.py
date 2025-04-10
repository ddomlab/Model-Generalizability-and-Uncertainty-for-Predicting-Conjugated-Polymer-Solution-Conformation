from pathlib import Path
import pandas as pd
from clean_dataset import main_cleaning, drop_block_cp, map_derived_Rh_data, Rh_columns_to_map

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
RAW_dir = DATASETS/ 'raw'
JSONS =  DATASETS/'json_resources'

raw_dataset: pd.DataFrame = pd.read_excel(RAW_dir/'Polymer_Solution_Scattering_Dataset.xlsx') 
Rh_data: pd.DataFrame = pd.read_pickle(RAW_dir/'Rh distribution-intensity weighted.pkl')



targets = [ 'Rh (IW avg log)', 'Rg1 (nm)', 'Lp (nm)']

data_summary_monitor = {
    'Target': [],
    'Initial Cleaned': [],
    'After dropping block copolymer': [],
    'After drop_block_cp': [],
    'After dropping  solid additives': [],
    'After dropping polymer HSPs': [],
    'Reduced After dropping block copolymer': [],
    'Reduced After dropping  solid additives': [],
    'Reduced After polymerdropping HSPs': []
}



if __name__ == "__main__":
    # print(Rh_data)
    # raw_dataset['intensity weighted average Rh (nm)'] = raw_dataset.index.to_series().apply(
    # map_intensity_weighted_rh, args=(Rh_data, raw_dataset,'intensity weighted average Rh (nm)'))
    
    # raw_dataset['intensity weighted average over log(Rh (nm))'] = raw_dataset.index.to_series().apply(
    # map_intensity_weighted_rh, args=(Rh_data, raw_dataset,'intensity weighted average over log(Rh (nm))'))

    mapped_dataset = map_derived_Rh_data(raw_dataset, Rh_data, Rh_columns_to_map)
    print("Done with mapping Rh data")
    cleaned_df = main_cleaning(mapped_dataset)
    Initial_counts = {t: cleaned_df[t].notna().sum() for t in targets}

    print("Done with cleaning")
    dropped_blocopolymer_data= drop_block_cp(cleaned_df)
    after_dropping_block_cp_counts = {t: dropped_blocopolymer_data[t].notna().sum() for t in targets}

    for t in targets:


        data_summary_monitor['Target'].append(t)
        data_summary_monitor['Initial Cleaned'].append(Initial_counts[t])
        data_summary_monitor['After dropping block copolymer'].append(after_dropping_block_cp_counts[t])
        data_summary_monitor['Reduced After dropping block copolymer'].append(Initial_counts[t] - after_dropping_block_cp_counts[t])


    print(data_summary_monitor)