import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from visualization_setting import set_plot_style, save_img_path
from typing import Callable, Optional, Union, Dict, Tuple
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import json
from visualize_ood_scores import get_score
set_plot_style(tick_size=16)

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



def weighted_jaccard(u, v,eps: float = 1e-6):
    min_sum = np.sum(np.minimum(u, v))
    max_sum = np.sum(np.maximum(u, v))
    return 1 - ((min_sum+eps) / (max_sum+eps))

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


sd_caler = StandardScaler()
cov_continuous_vector = rg_data[covarience_features].to_numpy(dtype=float)
cov_continuous_vector_scaled = sd_caler.fit_transform(cov_continuous_vector)
cov_mordred_vector = pd.DataFrame(rg_data['Trimer_Mordred'].tolist()).to_numpy(dtype=float)
cov_mordred_vector_scaled = sd_caler.fit_transform(cov_mordred_vector)
cov_ECFP_vector = np.array(rg_data['Trimer_ECFP6_count_512bits'].tolist())
cov_mordred_and_continuous_vector = np.concatenate([cov_mordred_vector, cov_continuous_vector], axis=1)
cov_mordred_and_continuous_vector_scaled = sd_caler.fit_transform(cov_mordred_and_continuous_vector)

naming: dict = {
        'mordred vector': cov_mordred_vector_scaled,
        'ECFP vector': cov_ECFP_vector,
        'numerical vector': cov_continuous_vector_scaled,
        'combined mordred-numerical vector': cov_mordred_and_continuous_vector_scaled,
}

results_path = HERE.parent.parent / 'results'/ 'OOD_target_log Rg (nm)'
cluster_types = 'KM5 polymer_solvent HSP and polysize cluster'
scores_folder_path = results_path / cluster_types/ 'Trimer_scaler'
# print(os.path.exists(scores_path))
score_file = scores_folder_path/ '(Mordred-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_NGB_Standard_scores.json'

def plot_OOD_Score_vs_distance(ml_score_metric:str, co_vector):
     
    with open(score_file, "r") as f:
        scores = json.load(f)

    clustering_score_metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]

    data = []
    for cluster_id in rg_data[cluster_types].unique():
        if cluster_id == 'rest':
            print('yes')
            continue
        labels = np.where(rg_data[cluster_types] == cluster_id, "test", "train")
        metric = weighted_jaccard if co_vector == 'ECFP vector' else 'euclidean'
        si_score, db_score, ch_score = get_cluster_scores(naming[co_vector], labels, metric=metric)

        cluster_data = {
            "Cluster": cluster_id,
            "Silhouette": si_score,
            "Davies-Bouldin": db_score,
            "Calinski-Harabasz": ch_score
        }
        
        mean, std = get_score(scores, f"CO_{cluster_id}", ml_score_metric)
        cluster_data[f"{ml_score_metric}_mean"] = mean
        cluster_data[f"{ml_score_metric}_std"] = std
        
        data.append(cluster_data)

    df = pd.DataFrame(data)

    num_clusters = df["Cluster"].nunique()
    palette = sns.color_palette("husl", num_clusters)
    color_map = {cluster: palette[i] for i, cluster in enumerate(sorted(df["Cluster"].unique()))}

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cluster], markersize=10, label=f"CO {cluster}")
        for cluster in sorted(df["Cluster"].unique())
    ]

    for clustering_metric in clustering_score_metrics:
        plt.figure(figsize=(8, 6))
        
        for _, row in df.iterrows():
            plt.errorbar(row[clustering_metric], row[f"{ml_score_metric}_mean"], 
                        yerr=row[f"{ml_score_metric}_std"], fmt='o', 
                        color=color_map[row["Cluster"]], capsize=4, capthick=1.2, markersize=10)

        plt.xlabel(f"Train-Test Distance ({clustering_metric})")
        plt.ylabel(f"{ml_score_metric.upper()} Score")
        plt.title(f"Cluster {ml_score_metric.upper()} vs. Train-Test Distance ({clustering_metric})")
        plt.legend(handles=legend_elements, loc="best", fontsize=16)
        plt.tight_layout()
        save_img_path(scores_folder_path/'score vs distance (NGB_Mordred_polysize_HSPs_solvent properties_Standard)', f"{ml_score_metric} vs ({clustering_metric}) using {co_vector}.png")
        plt.close()



if __name__ == "__main__":
    score_metrics = ["rmse", "r2"]
    co_vectors = ['numerical vector', 'mordred vector', 'combined mordred-numerical vector']
    for ml_metric in score_metrics:
        for co_vector in co_vectors:
            plot_OOD_Score_vs_distance(ml_metric, co_vector)
