import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
training_df_dir: Path = DATASETS/ "training_dataset"

working_data = pd.read_pickle(training_df_dir/"Rg_training_data.pkl")


unique_df = working_data.drop_duplicates(subset="canonical_name")
print(len(unique_df))
binary_vectors = np.array(unique_df["Monomer_ECFP12_binary_4096bits"].tolist())

# Step 2: Compute pairwise Tanimoto similarities
# `pdist` computes distances; subtract from 1 to get similarities
tanimoto_similarities = 1 - pdist(binary_vectors, metric="jaccard")
# Step 3: Visualize the distribution of Tanimoto similarities
plt.figure(figsize=(10, 6))
sns.histplot(tanimoto_similarities, bins=30, kde=True, color='blue')
plt.title("Distribution of Tanimoto Similarities for Unique Polymers in Rg dataset", fontsize=14)
plt.xlabel("Tanimoto Similarity", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

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