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
training_df_dir: Path = DATASETS/ "training_dataset"/ "dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended.pkl"
w_data = pd.read_pickle(training_df_dir)



def intensity_weighted_average_over_log(R_h, peaks, intensity):
    intensity_at_peaks = [intensity[i] for i in peaks]
    R_h = np.array(R_h)
    I = np.array(intensity_at_peaks)
    weighted_avg = np.sum(np.log10(R_h) * I) / np.sum(I)
    return 10**weighted_avg


def reorder_and_pad(values, peaks, intensity,l1,l2):
    distances = [0, 0, 0]  
    if values is None:
        return values, distances
    
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
            distances[key] = float(np.max(range_values) - np.min(range_values))
        elif len(range_values) == 1:
            result[key] = range_values[0]
            distances[key] = 0
    if peaks is None:
        for key, (lower, upper) in ranges.items():
            in_range = (values >= lower) & (values < upper)
            if in_range.any():
                result[key] = values[in_range][0] 
                distances[key] = 0
    return result, distances


def get_padding(value):
    if isinstance(value, list):
        # If the list is shorter than 3, pad with zeros
        if len(value) < 3:
            return value + [0] * (3 - len(value))
        # If the list is longer than 3, truncate to keep the smallest values
        elif len(value) > 3:
            return sorted(value)[:3]
        # If the list is already of length 3, return as is
        else:
            return value
    # Preserve NaN values
    elif pd.isna(value):
        return np.nan
    # Handle unexpected cases gracefully
    else:
        raise ValueError("The column contains unexpected non-list, non-NaN values.")
    

def plot_peak_distribution(data:pd.DataFrame, column_name:str,l1:int,l2:int):
    df= data.dropna(subset=[column_name])
    reordered_df = pd.DataFrame(df[column_name].tolist(), columns=["First Peak", "Second Peak", "Third Peak"])

    melted_df = reordered_df.melt(var_name="Peak Position", value_name="Value")
    melted_df['transformed_value'] = np.log10(melted_df['Value'])

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.histplot(data=melted_df, x="transformed_value", kde=True, hue="Peak Position",bins=40, ax=ax)

    ax.set_xlabel('log (Rh nm)',fontsize=20)
    ax.set_ylabel('Occurrence',fontsize=20)
    ax.set_title(f'Distribution of Rh Peak Values with padding',fontsize=30)
    ax.tick_params(axis='x', labelsize=25)  # Set font size for x-axis ticks
    ax.tick_params(axis='y', labelsize=25)  # Set font size for y-axis ticks
    box_inset = ax.inset_axes([0.01, -0.35, 0.99, 0.2])  
  
    sns.boxplot(x="transformed_value", data=melted_df, hue="Peak Position", ax=box_inset)
    box_inset.set(yticks=[], xlabel=None)
    box_inset.legend_.remove()
    box_inset.tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    save_img_path(VISUALIZATION/"analysis and test",f"Distribution of Rh Peak Values after padding.png")
    plt.close()



def plot_non_zero_counts(df:pd.DataFrame, column:str, num_indices:int=3):

    non_zero_counts = [
        df[column].apply(lambda x: x[i] != 0 if isinstance(x, list) and len(x) > i else False).sum()
        for i in range(num_indices)
    ]

    # Bar plot
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(1, num_indices + 1), non_zero_counts, color='skyblue', edgecolor='black')

    # Annotate the values above each bar
    for bar, count in zip(bars, non_zero_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count),
                 ha='center', va='bottom', fontsize=18)

    plt.xlabel('Peak Order', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.title(f'Non-Zero Counts at different peak orders with padding', fontsize=30)
    plt.xticks(range(1, num_indices + 1))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    save_img_path(VISUALIZATION/"analysis and test",f"Non-Zero Counts at different peak orders with padding.png")
    plt.close()



# def plot_violin_with_swarm(data, distance_column):

#     first_distances = [np.log10(d[0]) for d in data[distance_column] if d[0] is not None and d[0] > 0]
#     second_distances = [np.log10(d[1]) for d in data[distance_column] if d[1] is not None and d[1] > 0]
#     third_distances = [np.log10(d[2]) for d in data[distance_column] if d[2] is not None and d[2] > 0]
#     print("numer of instances in first peaks", len(first_distances))
#     print("numer of instances in second peaks", len(second_distances))
#     print("numer of instances in third peaks", len(third_distances))

    
#     plot_data = pd.DataFrame({
#         "Value": first_distances + second_distances + third_distances,
#         "peak_order": (["First"] * len(first_distances) +
#                       ["Second"] * len(second_distances) +
#                       ["Third"] * len(third_distances)),

#     })

#     plt.figure(figsize=(14, 7))
#     sns.violinplot(x="peak_order", y="Value", data=plot_data,
#                    split=True, palette="Set2")

#     sns.swarmplot(x="peak_order", y="Value", data=plot_data,
#                   dodge=True, color="k", alpha=0.6, size=4)

#     annotation_y = max(plot_data["Value"]) + 1  # Position above the maximum value
#     plt.annotate(f"{len(first_distances)} instances", xy=(0, annotation_y),
#                  xytext=(0, annotation_y + 0.1), ha='center', fontsize=20, color='blue')
#     plt.annotate(f"{len(second_distances)} instances", xy=(1, annotation_y),
#                  xytext=(1, annotation_y + 0.1), ha='center', fontsize=20, color='blue')
#     plt.annotate(f"{len(third_distances)} instances", xy=(2, annotation_y),
#                  xytext=(2, annotation_y + 0.1), ha='center', fontsize=20, color='blue')


#     plt.title(f"Distribution of distances between Rh in the same range after spliting (limits of {l1}-{l2} nm)", fontsize=20)
#     plt.xlabel("Peak order", fontsize=30)
#     plt.ylabel("Log10(distance)", fontsize=30)
#     plt.xticks(fontsize=25)
#     plt.yticks(fontsize=25)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     save_path(VISUALIZATION/"analysis and test",f"Distribution Rh distance (limits of {l1}-{l2} nm).png")
#     plt.close()


def expand_peaks(df:pd.DataFrame, column:str, zero_replacement:bool,new_columns:list):
    df_peaks = df[column].apply(lambda x: x if isinstance(x, list) or isinstance(x, np.ndarray) else [np.nan] * 3)
    df_peaks = pd.DataFrame(df_peaks.tolist(), columns=new_columns)
    if zero_replacement:
        df_peaks.replace(0, np.nan, inplace=True)
    df = pd.concat([df, df_peaks], axis=1)

    return df


if __name__ == "__main__":
    # for i1 in [40,50,70,80,90, 100]:
    # for i2 in [900,1000,1100,1200,1500, 1900,2000]:
        # l3=3500

        l1 = 40
        l2 = 1000
        w_data["multimodal Rh"], w_data["distances"] = zip(*w_data.apply(
        lambda row: reorder_and_pad(
            row["Rh at peaks (above 1 nm)"],
            row["peak index (above 1 nm)"],
            row["normalized intensity (0-1) corrected"],
            l1=l1,
            l2=l2,
        ),
        axis=1
        ))


        w_data["multimodal Rh with padding"] = w_data["Rh at peaks (above 1 nm)"].apply(get_padding)
    
        if "distances" in w_data.columns:
            w_data.drop(columns=["distances"], inplace=True)
        new_exanded_col_with_zero_replaced = ['First Peak_wo placeholder', 'Second Peak_wo placeholder', 'Third Peak_wo placeholder'] # dropped palce holder
        w_data = expand_peaks(w_data,"multimodal Rh", zero_replacement=True,new_columns=new_exanded_col_with_zero_replaced)
        # print(w_data['Third Peak'].notna().sum())
        w_data["multimodal Rh (e-5 place holder)"] = w_data["multimodal Rh"].apply(lambda x: [1e-5 if v == 0 else v for v in x] if isinstance(x, list) else x)

        w_data["log multimodal Rh (e-5 place holder)"] = w_data["multimodal Rh (e-5 place holder)"].apply(lambda x: np.log10(x) if isinstance(x, list) else x)
        new_exanded_col_with_new_place_holder = ['log First Peak (e-5 place holder)', 'log Second Peak (e-5 place holder)', 'log Third Peak (e-5 place holder)']
        w_data = expand_peaks(w_data,"log multimodal Rh (e-5 place holder)", zero_replacement=False,new_columns=new_exanded_col_with_new_place_holder)
        w_data[['log First Peak wo placeholder', 'log Second Peak wo placeholder', 'log Third Peak wo placeholder']] = w_data[new_exanded_col_with_zero_replaced].applymap(lambda x: np.log10(x) if x > 0 else None)
        # print(w_data[["multimodal Rh",'log First Peak (e-5 place holder)','log First Peak wo placeholder', 'log Second Peak wo placeholder', 'log Third Peak wo placeholder']])

        w_data.to_pickle(DATASETS/"training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal (40-1000 nm)_added.pkl")



        # plot_peak_distribution(w_data,"multimodal Rh with padding",l1,l2)
        # plot_non_zero_counts(w_data, "multimodal Rh with padding", 3)
        # plot_violin_with_swarm(w_data,"distances")
        # num_rows = w_data["Rh at peaks (above 1 nm)"].apply(
        #         lambda x: isinstance(x, list) and len(x) >= 3
        #     ).sum()
        # print(f"Number of rows with lists of length 3 or more: {num_rows}")


        
        # print(w_data)
        # zero_counts = w_data["multimodal Rh (e-5 place holder)"].apply(
        #     lambda x: [x[0] == 0 if isinstance(x, list) and len(x) > 0 else False,
        #             x[1] == 0 if isinstance(x, list) and len(x) > 1 else False,
        #             x[2] == 0 if isinstance(x, list) and len(x) > 2 else False]
        # ).apply(pd.Series).sum()

        # # Display results
        # print(f"Number of zeros in the first element: {zero_counts[0]}")
        # print(f"Number of zeros in the second element: {zero_counts[1]}")
        # print(f"Number of zeros in the third element: {zero_counts[2]}")