import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


HERE: Path = Path(__file__).resolve().parent
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

# Function to reorder and pad values
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
            # Find corresponding peaks and apply weighted averaging
            range_peaks = np.array(peaks)[in_range]
            result[key] = intensity_weighted_average_over_log(
                range_values.tolist(),
                range_peaks.tolist(),
                intensity
            )
        elif len(range_values) == 1:
            result[key] = range_values[0]

    # Handle cases where peaks are None or not applicable
    if peaks is None:
        for key, (lower, upper) in ranges.items():
            in_range = (values >= lower) & (values < upper)
            if in_range.any():
                result[key] = values[in_range][0]  # Assign the value directly

    return result


def plot_peak_distribution(data, column_name):
    # Reorder and pad values in the specified column
    df= data.dropna(subset=[column_name])
    reordered_df = pd.DataFrame(df[column_name].tolist(), columns=["First Peak", "Second Peak", "Third Peak"])

    # Melt the dataframe to long format for plotting
    melted_df = reordered_df.melt(var_name="Peak Position", value_name="Value")
    melted_df['transformed_value'] = np.log10(melted_df['Value'])
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with KDE
    sns.histplot(data=melted_df, x="transformed_value", kde=True, hue="Peak Position", ax=ax)

    # Customize the histogram
    ax.set_xlabel('log (Rh nm)')
    ax.set_ylabel('Occurrence')
    ax.set_title('Distribution of Peak Values')

    # Create inset boxplot
    box_inset = ax.inset_axes([0.01, -0.35, 0.99, 0.2])  # Adjust position for the inset box plot
  
    sns.boxplot(x="transformed_value", data=melted_df, hue="Peak Position", ax=box_inset)
    box_inset.set(yticks=[], xlabel=None)
    box_inset.legend_.remove()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    w_data["multimodal Rh"] = w_data.apply(
    lambda row: reorder_and_pad(
        row["Rh at peaks (above 1 nm)"],
        row["peak index (above 1 nm)"],
        row["normalized intensity (0-1) corrected"],
        l1=100,
        l2=1000,
        # threshold=100000
    ),
    axis=1
    )
    w_data.to_pickle(DATASETS/"training_dataset"/"dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal_added.pkl")
    plot_peak_distribution(w_data,"multimodal Rh")
