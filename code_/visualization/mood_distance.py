import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from visualization_setting import set_plot_style
from typing import Callable, Optional, Union, Dict, Tuple
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

set_plot_style()

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
training_df_dir: Path = DATASETS/ "training_dataset"

# working_data = pd.read_pickle(training_df_dir/'dataset_wo_block_cp_(fp-hsp)_added_additive_dropped_polyHSP_dropped_peaks_appended_multimodal (40-1000 nm)_added.pkl')

# Rg_data = working_data[working_data["Rg1 (nm)"].notna()]
# unique_df = Rg_data.drop_duplicates(subset="canonical_name")
# print(len(unique_df))
# binary_vectors = np.array(unique_df["Monomer_ECFP6_binary_512bits"].tolist())
# count_vectors = np.array(unique_df['Monomer_ECFP6_count_512bits'].tolist())

# binary_tanimoto_similarities = 1 - pdist(binary_vectors, metric="jaccard")



# def weighted_jaccard(u, v, eps:float=1e-6) -> float:
#     min_sum = np.sum(np.minimum(u, v))
#     max_sum = np.sum(np.maximum(u, v))
#     return 1 - ((min_sum + eps)/ (max_sum + eps)) 

# count_tanimoto_similarities = 1 - pdist(count_vectors, metric=weighted_jaccard)

# data = pd.DataFrame({
#     "Similarity": np.concatenate([binary_tanimoto_similarities, count_tanimoto_similarities]),
#     "Type": ["Binary"] * len(binary_tanimoto_similarities) + ["Count-based"] * len(count_tanimoto_similarities)
# })

# plt.figure(figsize=(9, 6))
# sns.histplot(data, x="Similarity", hue="Type", kde=True, bins=20)
# plt.title("Comparison of Binary and Count-based Tanimoto Similarities")
# plt.xlabel("Tanimoto Similarity")
# plt.ylabel("Frequency")
# plt.tight_layout()
# # visualization_folder_path =  HERE/"analysis and test"
# # os.makedirs(visualization_folder_path, exist_ok=True)    
# # fname = "Comparison of Binary and Count-based Tanimoto Similarities (over Rg)"
# # plt.savefig(visualization_folder_path / f"{fname}.png", dpi=600)
# # plt.close()

# plt.show()



## Train-Test OOD Distance

rg_data = pd.read_pickle(training_df_dir/'Rg data with clusters.pkl')
print(rg_data['EG-Ionic-Based Cluster'])


def get_cluster_scores(fp_vector:np.ndarray, predicted_clusters, metric: Union[str, Callable]):
    si_score = silhouette_score(fp_vector, predicted_clusters, metric=metric)
    db_score = davies_bouldin_score(fp_vector, predicted_clusters)
    ch_score = calinski_harabasz_score(fp_vector, predicted_clusters)
    return si_score, db_score, ch_score



covarience_features = ['Mw (g/mol)', 'PDI', 'Concentration (mg/ml)', 
                        'Temperature SANS/SLS/DLS/SEC (K)',
                        'polymer dP', 'polymer dD' , 'polymer dH',
                        'solvent dP', 'solvent dD', 'solvent dH']

clusters = ['KM4 ECFP6_Count_512bit cluster'	
'KM3 Mordred cluster'	
'substructure cluster'
'EG-Ionic-Based Cluster'
'KM5 polymer_solvent HSP and polysize cluster'	
'KM4 polymer_solvent HSP cluster'
'KM4 Mordred_Polysize cluster']


cov_continuous_vector = rg_data[covarience_features].to_numpy(dtype=float)
cov_mordred_vector = pd.DataFrame(rg_data['Trimer_Mordred'].tolist()).to_numpy(dtype=float)
cov_mordred_vector = np.array(rg_data['Trimer_ECFP6_count_512bits'].tolist())

mask

for 