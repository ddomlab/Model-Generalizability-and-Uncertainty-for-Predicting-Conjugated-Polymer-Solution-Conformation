from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
RAW_dir = DATASETS/ 'raw'
raw_rh_data = pd.read_csv(RAW_dir/'Rh distribution.csv')
rh_data = raw_rh_data[["index to extract",	"derived Rh (nm)",	"normalized intensity (0-1)"]].copy()


grouped_rh_data = rh_data.groupby("index to extract").agg({
    "derived Rh (nm)": lambda x: list(x),
    "normalized intensity (0-1)": lambda x: list(x)
}).reset_index()

scaler = MinMaxScaler()
grouped_rh_data['normalized intensity (0-1) corrected'] = grouped_rh_data['normalized intensity (0-1)'].apply(
    lambda x: scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten()
)


def intensity_weighted_average(R_h: np.ndarray, I: np.ndarray) -> float:
    # Ensure R_h and I are numpy arrays for element-wise operations
    R_h = np.array(R_h)
    I = np.array(I)

    # Intensity-weighted average of hydrodynamic radius
    weighted_avg = np.sum(R_h * I) / np.sum(I)

    return weighted_avg


def intensity_weighted_average_over_log(R_h: np.ndarray, I: np.ndarray) -> float:
    # Ensure R_h and I are numpy arrays for element-wise operations
    R_h = np.array(R_h)
    I = np.array(I)

    # Intensity-weighted average of hydrodynamic radius
    weighted_avg = np.sum(np.log10(R_h) * I) / np.sum(I)
    Rh_derived = 10**weighted_avg
    return Rh_derived

grouped_rh_data["intensity weighted average Rh (nm)"] = grouped_rh_data.apply(
        lambda row: intensity_weighted_average(row['derived Rh (nm)'], row['normalized intensity (0-1) corrected']),
    axis=1)


grouped_rh_data["intensity weighted average over log(Rh (nm))"] = grouped_rh_data.apply(
        lambda row: intensity_weighted_average_over_log(row['derived Rh (nm)'], row['normalized intensity (0-1) corrected']),
    axis=1)

# for idx, (rh_values, intensity_values) in enumerate(zip(grouped_rh_data['derived Rh (nm)'], grouped_rh_data['normalized intensity (0-1) corrected'])):
#     index_label = grouped_rh_data['index to extract'][idx]
#     plt.figure(figsize=(10, 6))

#     plt.plot(rh_values, intensity_values, marker='o', linestyle='-', color='b', label=index_label)
#     plt.xlabel("Rh (nm)")
#     plt.ylabel("Normalized Intensity (0-1)")
#     plt.title("Hydrodynamic Radius (Rh) vs. Normalized Intensity")
#     plt.yscale("linear")  # or "log" if you need a log scale for intensity
#     plt.xscale("log")  # assuming you want Rh on a log scale, as suggested by the plot you've shown
#     plt.legend()
#     plt.show()

if __name__ == "__main__":


    grouped_rh_data.to_pickle(RAW_dir/"Rh distribution-intensity weighted.pkl")
