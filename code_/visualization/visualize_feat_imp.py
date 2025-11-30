import json
from pathlib import Path
from typing import List, Optional, Any, Dict
import os 
import re

# import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from visualization_setting import set_plot_style, save_img_path, ensure_long_path
import shap
set_plot_style()

HERE = Path(__file__).resolve().parent
# DATASETS = HERE.parent.parent / "datasets" / "Validation datasets"
RESULTS = HERE.parent.parent / "results"




def plot_feature_importances(
    scores_data: Dict[str, Any],
    save_loc: Path,
    importance_type: str = "MDI",
    figsize: tuple = (10, 7),
    file_extension: str = ""
):
    """
    Plot top 15 feature importances.

    importance_type:
        "MDI"  -> model feature_importance_MDI (violin+swarm)
        "SHAP" -> full SHAP matrices (violin+swarm)
    """

    # ============================================================
    # 1. Extract MDI or SHAP importance
    # ============================================================
    if importance_type == "MDI":

        # Gather model importances
        if isinstance(scores_data.get("feature_importance_MDI"), list):
            all_MDI = scores_data["feature_importance_MDI"]
        else:
            all_MDI = []
            for v in scores_data.values():
                if isinstance(v, dict) and isinstance(v.get("feature_importance_MDI"), list):
                    all_MDI.extend(v["feature_importance_MDI"])

        if not all_MDI:
            raise ValueError("No model based importances found in scores_data.")

        df = pd.DataFrame(all_MDI)

        # Top 15 features by mean
        top_features = df.mean().sort_values(ascending=False).head(15).index
        df_long = df[top_features].melt(var_name="Feature", value_name="Value")

        # ============================================================
        # MDI violin + swarm plot
        # ============================================================
        plt.figure(figsize=figsize)

        sns.violinplot(
            data=df_long,
            x="Value",
            y="Feature",
            inner=None,
            cut=0,
            color="#298ed7",
        )

        sns.swarmplot(
            data=df_long,
            x="Value",
            y="Feature",
            size=3,
            alpha=0.7,
            color="#c64467",
        )

        plt.ylabel("Features")
        plt.xlabel("Model importance (MDI)")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        save_img_path(save_loc / "feature importance", f"{importance_type}_feature_importance_top15_{file_extension}.png")
        plt.close()

        return df, df_long


    # ============================================================
    # 2. SHAP: aggregate across all seeds, visualize both violin+swarm and SHAP summary
    # ============================================================
    elif importance_type == "SHAP":

        # ------------------------------------------------------------
        # Extract all SHAP matrices across seeds
        # ------------------------------------------------------------
        if isinstance(scores_data.get("feature_importance_SHAP_FULL"), list):
            shap_full = scores_data["feature_importance_SHAP_FULL"]
        else:
            shap_full = []
            for v in scores_data.values():
                if isinstance(v, dict) and isinstance(v.get("feature_importance_SHAP_FULL"), list):
                    shap_full.extend(v["feature_importance_SHAP_FULL"])

        if not shap_full:
            raise ValueError("No SHAP full matrices found.")

        feature_names = shap_full[0]["feature_names"]

        # ------------------------------------------------------------
        # Concatenate SHAP values and transformed X across all seeds
        # ------------------------------------------------------------
        all_shap = np.vstack([entry["shap_values"] for entry in shap_full])

        # Must be saved earlier in get_feature_importances_from_cv()
        all_Xt = np.vstack([np.array(entry["X_transformed"]) for entry in shap_full])


        # ------------------------------------------------------------
        # Build long dataframe for violin/swarm
        # ------------------------------------------------------------
        long_data = []
        for entry in shap_full:
            shap_matrix = np.asarray(entry["shap_values"])
            fn = entry["feature_names"]
            for feat_idx, feat in enumerate(fn):
                long_data.extend([[feat, val] for val in shap_matrix[:, feat_idx]])

        df_long = pd.DataFrame(long_data, columns=["Feature", "SHAP"])

        # Top 15 features by mean |SHAP|
        top_features = (
            df_long.groupby("Feature")["SHAP"]
            .apply(lambda x: np.mean(np.abs(x)))
            .sort_values(ascending=False)
            .head(15)
            .index
        )

        df_long = df_long[df_long["Feature"].isin(top_features)]
        df_long["Feature"] = pd.Categorical(df_long["Feature"], categories=top_features, ordered=True)

        # ------------------------------------------------------------
        # SHAP SUMMARY PLOT (ALL SEEDS MERGED)
        # ------------------------------------------------------------
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            all_shap,
            all_Xt,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()

        save_img_path(
            save_loc / "feature importance",
            f"SHAP_summary_{file_extension}.png",
        )
        plt.close()

        return None, df_long









# def krippendorff_alpha_by_feature(df, save_loc, file_extension, n_seeds=7, folds_per_seed=5, figsize=(12,6)):
#     total_needed = n_seeds * folds_per_seed

#     if len(df) < total_needed:
#         raise ValueError(f"Not enough rows: need {total_needed}, but got {len(df)}")

#     # Ensure we use exactly N rows (or you can shuffle before slicing)
#     df_cut = df.iloc[:total_needed]

#     alphas = {}

#     # Loop over all features
#     for feature in df_cut.columns:
#         values = df_cut[feature].values

#         # Reshape rows into (n_seeds × folds_per_seed)
#         ratings = values.reshape(n_seeds, folds_per_seed)

#         # Compute alpha
#         try:
#             alpha = krippendorff.alpha(
#                 reliability_data=ratings,
#                 level_of_measurement='interval'
#             )
#         except Exception:
#             alpha = np.nan

#         alphas[feature] = alpha

#     # Convert to DataFrame
#     alphas_df = pd.DataFrame({
#         "Feature": list(alphas.keys()),
#         "Alpha": list(alphas.values())
#     }).sort_values("Alpha", ascending=False)

#     # ---- Plot bar chart ----
#     plt.figure(figsize=figsize)
#     plt.bar(alphas_df["Feature"], alphas_df["Alpha"], color="#0b81a5")
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Krippendorff’s α")
#     # plt.title("Krippendorff’s Alpha for Each Feature")
#     plt.tight_layout()
#     save_img_path(save_loc / "feature importance", f"feature_krippendorff_stability_{file_extension}.png")
#     # plt.show()
#     plt.close()
#     return alphas_df


# def calculate_kendalls_w(df_input):
#     m = len(df_input)
#     n = len(df_input.columns)
    
#     df_ranked = df_input.rank(axis=1, ascending=False, method='average')
    
#     R = df_ranked.sum(axis=0)
    
#     R_bar_expected = m * (n + 1) / 2
    
#     S = np.sum((R - R_bar_expected)**2)
    
#     S_max = (m**2 * (n**3 - n)) / 12
    
#     W = S / S_max
    
#     Chi_sq = m * (n - 1) * W
#     df_chi_sq = n - 1
    
#     return {
#         "m (Runs)": m,
#         "n (Features)": n,
#         "S": S,
#         "S_max": S_max,
#         "Kendall's W": W,
#         "Chi-square": Chi_sq,
#         "Degrees of Freedom": df_chi_sq
#     }



if __name__ == "__main__":

    PAPER = {
            "Robust Learning from Literature Data_Model Generalizability and Uncertainty for Predicting Conjugated Polymer Solution Conformation": ["target_log Rg (nm)"],
            # "Beyond molecular structure_ critically assessing machine learning for designing organic photovoltaic materials and devices": ["target_calculated PCE (%)"],
            # "Machine Learning for Polymer Design to Enhance Pervaporation-Based Organic Recovery": ["target_log (Separation factor)","target_log (Total flux)"],
            # "Machine Learning-Enabled Prediction and High-Throughput Screening of Polymer Membranes for Pervaporation Separation": ["target_log (Separation factor)","target_log (Total flux)"],
            # "Understanding and Designing a High-Performance Ultrafiltration Membrane Using Machine Learning": [
            # "target_flux decline ratio (%)",
            # "target_flux recovery ratio (%)",
            # "target_irreversible fouling ratio(%)",
            # "target_organic compound removal (%)",
            # "target_reversible fouling ratio (%)",
            # r"target_water permeability (LMH\bar)",
            # ],
            }
    
    models = ["RF","XGBR"]

    for model in models:
        paper_loc: Path = Path(r"D:\PhD_Code\Model-Generalizability-and-Uncertainty-for-Predicting-Conjugated-Polymer-Solution-Conformation\results\target_log Rg (nm)\scaler")
        file_name = f"(Xn-Mw-PDI-concentration-temperature-polymer dP-polymer dD-polymer dH-solvent dP-solvent dD-solvent dH-light exposure-aging time-aging temperature-prep temperature-prep time-model_fitting_encoded)_{model}_hypOFF_Standard_FeatImp_scores"
        score_path = ensure_long_path(paper_loc / f"{file_name}.json")
        with open(score_path, "r") as f:
            scores = json.load(f)

        MDI_imp, shap_imp = plot_feature_importances(scores_data=scores,
                                        save_loc=paper_loc.parent,
                                        figsize=(8,7.5),
                                        importance_type="MDI",
                                        file_extension=f"all_nums_fitting_models_included_{model}"
                                        )
    #             shap_feature_means = shap_imp.abs().mean()
    #             df_top15_shap_features = shap_imp[shap_feature_means.sort_values(ascending=False).head(15).index]

    #             mdi_feature_means = MDI_imp.mean()
    #             df_top15_mdi_features = MDI_imp[mdi_feature_means.sort_values(ascending=False).head(15).index]

    #             # 3. Filter the DataFrame to these 15 features
    #             # print(df_top15)
    #             # plot_top15_feature_stability(
    #             #                     scores_data=scores,
    #             #                     # save_loc=paper_loc,
    #             #                     # file_extension=file_name,
    #             #                     # top_n=15,
    #             #                     # figsize=(8,6)
    #             #                     )
    #             # print(df_top15)
    #             # krippendorff_alpha_by_feature(
    #             #                             df=df_top15,             
    #             #                             save_loc=paper_loc,
    #             #                             file_extension=file_name,
    #             #                             figsize=(9,6)
    #             #                             )
    #             # print(calculate_kendalls_w(df_top15))
    #             model_stats.setdefault(paper_name, {}).setdefault(target, {}).setdefault(model, {})["SHAP"] = calculate_kendalls_w(df_top15_shap_features)
    #             model_stats.setdefault(paper_name, {}).setdefault(target, {}).setdefault(model, {})["MDI"] = calculate_kendalls_w(df_top15_mdi_features)
    #             # print(pg.friedman(df_top15))

    # with open(RESULTS / "model_stats" / "model_stability.json", "w") as f:
    #     json.dump(model_stats, f, indent=2)