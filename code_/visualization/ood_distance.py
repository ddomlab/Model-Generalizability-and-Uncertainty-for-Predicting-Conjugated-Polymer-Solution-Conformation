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
from matplotlib.lines import Line2D
from visualize_ood_scores import get_score, ensure_long_path, process_learning_curve_scores, plot_bar_ood_iid
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

## Train-Test OOD Distance

rg_data = pd.read_pickle(training_df_dir/'Rg data with clusters.pkl')
results_path = HERE.parent.parent / 'results'/ 'OOD_target_log Rg (nm)'

def get_cluster_scores(fp_vector:np.ndarray, predicted_clusters, metric: Union[str, Callable]):
    si_score = silhouette_score(fp_vector, predicted_clusters, metric=metric)
    db_score = davies_bouldin_score(fp_vector, predicted_clusters)
    ch_score = calinski_harabasz_score(fp_vector, predicted_clusters)
    return si_score, db_score, ch_score



covarience_features = ['Xn', 'Mw (g/mol)', 'PDI', 'Concentration (mg/ml)', 
                        'Temperature SANS/SLS/DLS/SEC (K)',
                        'polymer dP', 'polymer dD' , 'polymer dH',
                        'solvent dP', 'solvent dD', 'solvent dH']


sd_caler = StandardScaler()
cov_continuous_vector = rg_data[covarience_features].to_numpy(dtype=float)
cov_continuous_vector_scaled = sd_caler.fit_transform(cov_continuous_vector)
cov_mordred_vector = pd.DataFrame(rg_data['Trimer_Mordred'].tolist()).to_numpy(dtype=float)
cov_mordred_vector_scaled = sd_caler.fit_transform(cov_mordred_vector)
cov_ECFP_vector = np.array(rg_data['Trimer_ECFP6_count_512bits'].tolist(), dtype=int)
cv_MACCS_vector = np.array(rg_data['Trimer_MACCS'].tolist())
cov_mordred_and_continuous_vector = np.concatenate([cov_mordred_vector, cov_continuous_vector], axis=1)
cov_mordred_and_continuous_vector_scaled = sd_caler.fit_transform(cov_mordred_and_continuous_vector)

naming: dict = {
        'mordred vector': cov_mordred_vector_scaled,
        'ECFP vector': cov_ECFP_vector,
        'numerical vector': cov_continuous_vector_scaled,
        'combined mordred-numerical vector': cov_mordred_and_continuous_vector_scaled,
        'MACCS vector': cv_MACCS_vector,
}


def get_ood_iid(scores,ml_score_metric: str,algorithm: str):
    data = []
    for cluster, _ in scores.items():
        if cluster.startswith("CO_") or cluster.startswith("ID_") or cluster.startswith("IID_"):
            # print(cluster)
            # cluster_id = cluster.split("_")[1]
            mean_ood, std_ood = get_score(scores, f"{cluster}", ml_score_metric)
            data.append({
            "Cluster": cluster,
            "Model": algorithm,
            f"Score": mean_ood,
            f"Std": std_ood,
        })
    return data



def make_accumulating_scores(scores, ml_score_metric: str, 
                             co_vector, cluster_types: str,
                            algorithm: str, is_equal_size:bool=False,
                            is_ood_iid_distance:bool=True) -> list:
    if 'ECFP' in co_vector:
        metric = weighted_jaccard 
    elif 'MACCS' in co_vector:
        metric = 'jaccard'
    else:
        metric = 'euclidean'

    data = []
    all_clusters = rg_data[cluster_types].unique().tolist()

    if cluster_types == 'substructure cluster':
        all_clusters.append('Polar')

    for cluster_id in all_clusters:
        labels = (
            np.where(rg_data['Side Chain Cluster'] == 'Polar', "test", "train")
            if cluster_id == 'Polar'
            else np.where(rg_data[cluster_types] == cluster_id, "test", "train")
        )

        si_score, db_score, ch_score = get_cluster_scores(naming[co_vector], labels, metric=metric)

        if is_equal_size==True:
            _, scores_at_equal_training_size = process_learning_curve_scores(scores,ml_score_metric)
            cluster_scores = scores_at_equal_training_size[scores_at_equal_training_size['Cluster'] == f"CO_{cluster_id}"]
            cluster_test_scores = cluster_scores[cluster_scores["Score Type"] == "Test"]
            mean = cluster_test_scores["Score"].iloc[0]
            std = cluster_test_scores["Std"].iloc[0]
        else:
            if is_ood_iid_distance==True:
                mean_ood, std_ood = get_score(scores, f"CO_{cluster_id}", ml_score_metric)
                mean_ID, std_ID = get_score(scores, f"ID_{cluster_id}", ml_score_metric)
                mean = abs(mean_ood - mean_ID)
                std = std_ood + std_ID
            else:
                mean_ood, std_ood = get_score(scores, f"CO_{cluster_id}", ml_score_metric)
                mean = mean_ood
                std = std_ood
        data.append({
            "Cluster": cluster_id,
            "Model": algorithm,
            "Silhouette": si_score,
            "Davies-Bouldin": db_score,
            "Calinski-Harabasz": ch_score,
            f"{ml_score_metric}_mean": mean,
            f"{ml_score_metric}_std": std
        })
    return data


marker_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X', 'h']

def plot_OOD_Score_vs_distance(df, ml_score_metric: str, co_vector,
                                saving_path: Optional[Path] = None, file_name: str = None,
                                is_equal_size:bool=False,
                                is_ood_iid_distance:bool=True) -> None:
    
    clustering_score_metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
    models = list({row['Model'] for row in df})
    clusters = list({row['Cluster'] for row in df})

    palette = sns.color_palette("Set2", len(clusters))
    color_map = {cluster: palette[i] for i, cluster in enumerate(sorted(clusters))}
    marker_map = {model: marker_shapes[i % len(marker_shapes)] for i, model in enumerate(sorted(models))}

    legend_elements = [
            Line2D([0], [0], marker='_', color=color_map[cluster], linestyle='None',
                   label=f"CO {cluster}", markersize=20, markeredgewidth=2)
            for cluster in sorted(clusters)
        ] + [
            Line2D([0], [0], marker=marker_map[model], color='k', linestyle='None', 
                   label=model, markersize=10)
            for model in sorted(models)
        ]

    for clustering_metric in clustering_score_metrics:
        plt.figure(figsize=(6, 5))

        for row in df:
            plt.errorbar(
                row[clustering_metric],
                row[f"{ml_score_metric}_mean"],
                yerr=row[f"{ml_score_metric}_std"],
                fmt=marker_map[row["Model"]],
                color=color_map[row["Cluster"]],
                capsize=4,
                capthick=1.2,
                markersize=10
            )

        if is_equal_size==True or is_ood_iid_distance==False:
            y_label = f"{ml_score_metric.upper()} Score"
        else:
            y_label = f"{ml_score_metric.upper()} Score (OOD-IID)"

        plt.ylabel(y_label, fontsize=16, fontweight='bold')
        plt.xlabel(f"Train-Test Distance ({clustering_metric})", fontsize=16, fontweight='bold')
        plt.title(f"{co_vector}".capitalize(), fontsize=20)
        plt.legend(handles=legend_elements, loc="best", fontsize=12)

        if clustering_metric.lower() == "silhouette":
            min_val = 0
            max_val = max(row[clustering_metric] for row in df)
            value_range = max_val - min_val
            step = 0.05 if value_range <= 0.2 else 0.1
            ticks = np.arange(min_val, round(max_val + step, 2), step)
            plt.xticks(ticks)
            plt.xlim(left=min_val)


        y_min = min(row[f"{ml_score_metric}_mean"] for row in df)
        y_max = max(row[f"{ml_score_metric}_mean"] for row in df)

        y_min_tick = np.floor(y_min * 5) / 5  # nearest lower multiple of 0.2
        y_max_tick = np.ceil(y_max * 5) / 5    # nearest higher multiple of 0.2
        yticks = np.arange(y_min_tick, y_max_tick + 0.2, 0.2)
        plt.yticks(yticks)
        plt.ylim(y_min_tick-.2, y_max_tick + 0.2)
        
        plt.tight_layout()
        save_img_path(saving_path, f"{file_name}_distance_metric-{clustering_metric}.png")
        # plt.show()
        plt.close()



if __name__ == "__main__":
    cluster_list = [
                    # 'KM4 ECFP6_Count_512bit cluster',	
                    'KM3 Mordred cluster',
                    'HBD3 MACCS cluster',
                    'substructure cluster',
                    # 'KM5 polymer_solvent HSP and polysize cluster',
                    # 'KM4 polymer_solvent HSP and polysize cluster',
                    'KM4 polymer_solvent HSP cluster',
                    'KM4 Mordred_Polysize cluster',
                    ]
    for cluster in cluster_list:
        score_metrics = ["rmse"]
        for accuracy_metric in score_metrics:
            ## Plot OOD-IID vs distance for mix of fingerprints and numerical

            # for fp in ['MACCS', 'Mordred', 'ECFP3.count.512']:
            #     co_vectors = ['numerical vector']
            #     if fp == 'Mordred':
            #         co_vectors.extend(['mordred vector', 'combined mordred-numerical vector'])
            #     if fp == 'MACCS':
            #         co_vectors.append('MACCS vector')

            #     if fp == 'ECFP3.count.512':
            #         co_vectors.append('ECFP vector')
                
            #     for co_vector in co_vectors:
            #         combined_data = []
            #         for model in ['XGBR', 'NGB', 'RF']:
            #                 scores_folder_path = results_path / cluster / 'Trimer_scaler'
            #                 score_file = scores_folder_path / f'({fp}-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_Standard_scores.json'
            #                 score_file = ensure_long_path(score_file)
            #                 if not os.path.exists(score_file):
            #                     print(f"File not found: {score_file}")
            #                     continue 

            #                 with open(score_file, "r") as f:
            #                     scores = json.load(f)

            #                 model_data = make_accumulating_scores(scores, accuracy_metric, co_vector, cluster, model,is_ood_iid_distance=False)
            #                 combined_data.extend(model_data)
                                    
            #         saving_folder = scores_folder_path/ f'scores vs distance combined (absolute)'/ f"{co_vector}"
            #         plot_OOD_Score_vs_distance(combined_data, accuracy_metric, co_vector=co_vector,
            #                         saving_path=saving_folder, file_name=f"fingerprint-{fp}_metric-{accuracy_metric}",is_ood_iid_distance=False)
            #         print('Plot combined')
                            
                    #  Plot OOD-IID vs distance for equal training size
                    #         score_file = scores_folder_path / f'({fp}-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_hypOFF_Standard_lc_scores.json'
                    #         score_file = ensure_long_path(score_file)
                    #         if not os.path.exists(score_file):
                    #             print(f"File not found: {score_file}")
                    #             continue 

                    #         with open(score_file, "r") as f:
                    #             scores = json.load(f)

                    #         model_data = make_accumulating_scores(scores, accuracy_metric, co_vector, cluster, model,is_equal_size=True)
                    #         combined_data.extend(model_data)


                    # saving_folder = scores_folder_path/ f'scores vs distance at equal training size'/ f"{co_vector}"
                    # plot_OOD_Score_vs_distance(combined_data, accuracy_metric, co_vector=co_vector,
                    #                 saving_path=saving_folder, file_name=f"fingerprint-{fp}_metric-{accuracy_metric}", is_equal_size=True)
                    # print('Plot equal size scores')




            ## Plot OOD-IID vs distance for only numerical
            # co_vector = 'numerical vector'
            
            # combined_data = []
            # for model in ['XGBR', 'NGB', 'RF']:
            #     scores_folder_path = results_path / cluster / 'Trimer_scaler'
            #     score_file = scores_folder_path / f'(MACCS-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_Standard_scores.json'
            #     score_file = ensure_long_path(score_file)
            #     with open(score_file, "r") as f:
            #         scores = json.load(f)
                    
            #     model_data = make_accumulating_scores(scores, accuracy_metric, co_vector, cluster, model,is_ood_iid_distance=False)
            #     combined_data.extend(model_data)

            # saving_folder = scores_folder_path/  f'scores vs distance combined (absolute)'/ f"{co_vector}"
            # plot_OOD_Score_vs_distance(combined_data, accuracy_metric, co_vector=co_vector,
            #                 saving_path=saving_folder, file_name=f"numerical_metric-{accuracy_metric}", is_ood_iid_distance=False)
            # print('Plot combined')

            #  Plot OOD-IID vs distance for equal training size
                # score_file = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_hypOFF_Standard_lc_scores.json'
                # score_file = ensure_long_path(score_file)
                # if not os.path.exists(score_file):
                #     print(f"File not found: {score_file}")
                #     continue 

                # with open(score_file, "r") as f:
                #     scores = json.load(f)

                # model_data = make_accumulating_scores(scores, accuracy_metric, co_vector, cluster, model,is_equal_size=True)
                # combined_data.extend(model_data)


            # saving_folder = scores_folder_path/ f'scores vs distance at equal training size'/ f"{co_vector}"
            # plot_OOD_Score_vs_distance(combined_data, accuracy_metric, co_vector=co_vector,
            #                 saving_path=saving_folder, file_name=f"numerical_metric-{accuracy_metric}", is_equal_size=True)
            # print('Plot equal size scores')
            

            ## plot bar plot for OOD-IID
            for fp in ['MACCS', 'Mordred', 'ECFP3.count.512']:
                combined_data = []
                for model in ['XGBR', 'NGB', 'RF']:
                    scores_folder_path = results_path / cluster / 'Trimer_scaler'
                    score_file = scores_folder_path / f'(MACCS-Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_{model}_Standard_scores.json'
                    score_file = ensure_long_path(score_file)
                    with open(score_file, "r") as f:
                        scores = json.load(f)
                    model_data = get_ood_iid(scores, accuracy_metric, model)
                    combined_data.extend(model_data)
                    print(pd.DataFrame(combined_data))
                saving_folder = scores_folder_path/  f'OOD-IID bar plot for full training set'
                plot_bar_ood_iid(pd.DataFrame(combined_data), accuracy_metric,
                                saving_folder,f'numerical-{fp}_metric-{accuracy_metric}', 
                                figsize=(8, 6), text_size=16,)

        # co_vectors = 'numerical vector'
        # score_metrics = ["rmse", "r2"]
        # for ml_metric in score_metrics:
        #     scores_folder_path = results_path / cluster / 'scaler'
        #     score_file = scores_folder_path / f'(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH)_RF_Standard_scores.json'
        #     score_file = ensure_long_path(score_file)
        #     with open(score_file, "r") as f:
        #         scores = json.load(f)
        #     saving_folder = scores_folder_path / f'scores vs distance'
        #     plot_OOD_Score_vs_distance(scores, ml_metric, co_vector=co_vectors,cluster_types=cluster,
        #                                 saving_path=saving_folder, file_name=f"RF_{co_vectors}_{ml_metric}")