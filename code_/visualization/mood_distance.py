import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
training_df_dir: Path = DATASETS/ "training_dataset"

working_data = pd.read_pickle(training_df_dir/"Rg_training_data.pkl")


unique_df = working_data.drop_duplicates(subset="canonical_name")
print(len(unique_df))
binary_vectors = np.array(unique_df["Monomer_ECFP12_binary_4096bits"].tolist())
count_vectors = np.array(unique_df["Monomer_ECFP12_count_4096bits"].tolist())

binary_tanimoto_similarities = 1 - pdist(binary_vectors, metric="jaccard")



def weighted_jaccard(u, v):
    min_sum = np.sum(np.minimum(u, v))
    max_sum = np.sum(np.maximum(u, v))
    return 1 - (min_sum / max_sum) if max_sum != 0 else 0

count_tanimoto_similarities = 1 - pdist(count_vectors, metric=weighted_jaccard)

data = pd.DataFrame({
    "Similarity": np.concatenate([binary_tanimoto_similarities, count_tanimoto_similarities]),
    "Type": ["Binary"] * len(binary_tanimoto_similarities) + ["Count-based"] * len(count_tanimoto_similarities)
})

sns.histplot(data, x="Similarity", hue="Type", kde=True, bins=20)
plt.title("Comparison of Binary and Count-based Tanimoto Similarities")
plt.xlabel("Tanimoto Similarity")
plt.ylabel("Frequency")
plt.tight_layout()
visualization_folder_path =  HERE/"analysis and test"
os.makedirs(visualization_folder_path, exist_ok=True)    
fname = "Comparison of Binary and Count-based Tanimoto Similarities (over Rg)"
plt.savefig(visualization_folder_path / f"{fname}.png", dpi=600)
plt.close()

   





# unique_df = working_data.drop_duplicates(subset="canonical_name")
# binary_vectors = np.array(unique_df["Monomer_ECFP12_binary_4096bits"].tolist())
# canonical_names = unique_df["canonical_name"].values  # Corresponding polymer names

# # Convert pairwise distances to a full similarity matrix
# distances = pdist(binary_vectors, metric="jaccard")
# tanimoto_similarities = 1 - distances
# similarity_matrix = squareform(tanimoto_similarities)

# # Step 3: Find pairs with Tanimoto similarity of 1
# def get_():

#     pairs = []
#     for i in range(similarity_matrix.shape[0]):
#         for j in range(i + 1, similarity_matrix.shape[0]):  # Upper triangle of the matrix
#             if similarity_matrix[i, j] == 1.0:
#                 pairs.append((canonical_names[i], canonical_names[j]))

#     # Step 4: Display results
#     if pairs:
#         print(f"Pairs of polymers with Tanimoto similarity of 1:\n{pairs}")
#     else:
#         print("No pairs of polymers have a Tanimoto similarity of 1.")