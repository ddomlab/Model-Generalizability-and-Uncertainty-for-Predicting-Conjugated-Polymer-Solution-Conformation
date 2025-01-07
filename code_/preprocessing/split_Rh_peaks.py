import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

HERE: Path = Path(__file__).resolve().parent
VISUALIZATION = HERE.parent/ "visualization"
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended.pkl"

w_data = pd.read_pickle(training_df_dir)



def intensity_weighted_average_over_log(R_h, peaks, intensity):
    intensity_at_peaks = [intensity[i] for i in peaks]
    R_h = np.array(R_h)
    I = np.array(intensity_at_peaks)
    weighted_avg = np.sum(np.log10(R_h) * I) / np.sum(I)
    return 10**weighted_avg

def reorder_and_pad(values, peaks, intensity,l1,l2):
    if values is None:
        return values
    
    values = np.array(values)
    ranges = {
        0: (0, l1),
        1: (l1, l2),
        2: (l2, float('inf'))
    }

    result = [0, 0, 0]

    for key, (lower, upper) in ranges.items():
        in_range = (values >= lower) & (values < upper)
        range_values = values[in_range]

        if len(range_values) > 1 and peaks is not None:
            range_peaks = np.array(peaks)[in_range]
            result[key] = intensity_weighted_average_over_log(
                range_values.tolist(),
                range_peaks.tolist(),
                intensity
            )
        elif len(range_values) == 1:
            result[key] = range_values[0]

    if peaks is None:
        for key, (lower, upper) in ranges.items():
            in_range = (values >= lower) & (values < upper)
            if in_range.any():
                result[key] = values[in_range][0] 

    return result


# def plot_peak_distribution(data:pd.DataFrame, column_name:str,l1:int,l2:int):
#     df= data.dropna(subset=[column_name])
#     reordered_df = pd.DataFrame(df[column_name].tolist(), columns=["First Peak", "Second Peak", "Third Peak"])

#     melted_df = reordered_df.melt(var_name="Peak Position", value_name="Value")
#     melted_df['transformed_value'] = np.log10(melted_df['Value'])

#     fig, ax = plt.subplots(figsize=(10, 6))

#     sns.histplot(data=melted_df, x="transformed_value", kde=True, hue="Peak Position",bins=40, ax=ax)

#     ax.set_xlabel('log (Rh nm)')
#     ax.set_ylabel('Occurrence')
#     ax.set_title(f'Distribution of Rh Peak Values (limits of {l1}-{l2})')

#     box_inset = ax.inset_axes([0.01, -0.35, 0.99, 0.2])  
  
#     sns.boxplot(x="transformed_value", data=melted_df, hue="Peak Position", ax=box_inset)
#     box_inset.set(yticks=[], xlabel=None)
#     box_inset.legend_.remove()
#     plt.tight_layout()
#     visualization_folder_path =  VISUALIZATION/"analysis and test"
#     os.makedirs(visualization_folder_path, exist_ok=True)    
#     fname = f"Distribution of Rh Peak Values after spliting (limits of {l1}-{l2} nm).png"
#     plt.savefig(visualization_folder_path / fname, dpi=600)
#     plt.close()

def reorder_and_pad(values, peaks, intensity, l1, l2, l3):
    if values is None:
        return values

    values = np.array(values)
    ranges = {
        0: (0, l1),
        1: (l1, l2),
        2: (l2, l3),
        3: (l3, float('inf'))
    }

    result = [0, 0, 0, 0]  # Initialize result for 4 ranges

    for key, (lower, upper) in ranges.items():
        in_range = (values >= lower) & (values < upper)
        range_values = values[in_range]

        if len(range_values) > 1 and peaks is not None:
            range_peaks = np.array(peaks)[in_range]
            result[key] = intensity_weighted_average_over_log(
                range_values.tolist(),
                range_peaks.tolist(),
                intensity
            )
        elif len(range_values) == 1:
            result[key] = range_values[0]

    if peaks is None:
        for key, (lower, upper) in ranges.items():
            in_range = (values >= lower) & (values < upper)
            if in_range.any():
                result[key] = values[in_range][0]

    return result


def plot_peak_distribution(data: pd.DataFrame, column_name: str, l1: int, l2: int, l3: int):
    df = data.dropna(subset=[column_name])
    reordered_df = pd.DataFrame(df[column_name].tolist(), columns=["First Peak", "Second Peak", "Third Peak", "Fourth Peak"])

    melted_df = reordered_df.melt(var_name="Peak Position", value_name="Value")
    melted_df['transformed_value'] = np.log10(melted_df['Value'])

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=melted_df, x="transformed_value", kde=True, hue="Peak Position", bins=40, ax=ax)

    ax.set_xlabel('log (Rh nm)')
    ax.set_ylabel('Occurrence')
    ax.set_title(f'Distribution of Rh Peak Values (limits of {l1}-{l2}-{l3})')

    box_inset = ax.inset_axes([0.01, -0.35, 0.99, 0.2])

    sns.boxplot(x="transformed_value", data=melted_df, hue="Peak Position", ax=box_inset)
    box_inset.set(yticks=[], xlabel=None)
    box_inset.legend_.remove()
    plt.tight_layout()
    
    visualization_folder_path = VISUALIZATION / "analysis and test"
    os.makedirs(visualization_folder_path, exist_ok=True)
    fname = f"Distribution of Rh Peak Values after splitting (limits of {l1}-{l2}-{l3} nm).png"
    plt.savefig(visualization_folder_path / fname, dpi=600)
    plt.close()


def plot_non_zero_counts(df:pd.DataFrame, column:str, num_indices:int=3):
    """
    Plots the count of non-zero elements at specified indices in a list column of a DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column: str, the name of the column containing lists.
    - num_indices: int, the number of indices to check in the lists.

    Returns:
    - None (displays a bar plot).
    """
    non_zero_counts = [
        df[column].apply(lambda x: x[i] != 0 if isinstance(x, list) and len(x) > i else False).sum()
        for i in range(num_indices)
    ]

    # Bar plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(1, num_indices + 1), non_zero_counts, color='skyblue', edgecolor='black')

    # Annotate the values above each bar
    for bar, count in zip(bars, non_zero_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count),
                 ha='center', va='bottom', fontsize=18)

    plt.xlabel('Peak Order', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Non-Zero Counts at different peak orders (limits of {l1}-{l2}-{l3} nm)', fontsize=16)
    plt.xticks(range(1, num_indices + 1))
    plt.tight_layout()
    visualization_folder_path =  VISUALIZATION/"analysis and test"
    os.makedirs(visualization_folder_path, exist_ok=True)    
    fname = f"Non-Zero Counts at different peak orders (limits of {l1}-{l2}-{l3} nm).png"
    plt.savefig(visualization_folder_path / fname, dpi=600)
    plt.close()

# Example usage





if __name__ == "__main__":
    # for i in [800,900,1000,1500,2500, 3000]:
    
        l1 = 30
        l2 = 1000
        l3=3500
        w_data["multimodal Rh"] = w_data.apply(
        lambda row: reorder_and_pad(
            row["Rh at peaks (above 1 nm)"],
            row["peak index (above 1 nm)"],
            row["normalized intensity (0-1) corrected"],
            l1=l1,
            l2=l2,
            l3=l3
            # threshold=100000
        ),
        axis=1
        )
        
        # w_data.to_pickle(DATASETS/"training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal_added.pkl")
        plot_peak_distribution(w_data,"multimodal Rh",l1,l2,l3)
        plot_non_zero_counts(w_data, 'multimodal Rh', 4)

        # print(w_data)
