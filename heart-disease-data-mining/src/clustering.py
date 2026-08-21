"""
Unsupervised patient risk segmentation using two different clustering
algorithms, compared head to head:
1. K-Means (centroid-based, assumes roughly spherical clusters)
2. Agglomerative Hierarchical Clustering (connectivity-based, no sphericity assumption)

We keep whichever produces the larger separation in actual CHD prevalence
between clusters, since that's the practically useful outcome for risk
segmentation -- and report both so the comparison itself is visible.
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import sys
sys.path.insert(0, "src")
from preprocessing import load_raw, clean, engineer_features, FEATURE_COLS
from sklearn.preprocessing import StandardScaler

FIG_DIR = "results/figures"
df = engineer_features(clean(load_raw()))
X = df[FEATURE_COLS]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]

# --- K-Means: choose k by silhouette score ---
inertias, sil_scores_km = [], []
k_range = range(2, 7)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores_km.append(silhouette_score(X_scaled, labels))

best_k = list(k_range)[int(np.argmax(sil_scores_km))]
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = km_final.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("K-Means Elbow")
axes[1].plot(list(k_range), sil_scores_km, marker="o", color="darkorange")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score"); axes[1].set_title("K-Means Silhouette")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/clustering_k_selection.png", dpi=150)
plt.close()

# --- Hierarchical clustering: same k for a fair comparison, plus its own silhouette ---
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
df["Hier_Cluster"] = agg.fit_predict(X_scaled)
sil_hier = silhouette_score(X_scaled, df["Hier_Cluster"])

# --- Compare prevalence spread between the two methods ---
km_prevalence = df.groupby("KMeans_Cluster")["TenYearCHD"].mean() * 100
hier_prevalence = df.groupby("Hier_Cluster")["TenYearCHD"].mean() * 100
km_spread = km_prevalence.max() - km_prevalence.min()
hier_spread = hier_prevalence.max() - hier_prevalence.min()

print(f"K-Means (k={best_k}): silhouette={max(sil_scores_km):.3f}, prevalence spread={km_spread:.1f}pp")
print(km_prevalence.round(1))
print(f"\nHierarchical (k={best_k}): silhouette={sil_hier:.3f}, prevalence spread={hier_spread:.1f}pp")
print(hier_prevalence.round(1))

winner = "K-Means" if km_spread >= hier_spread else "Hierarchical"
winning_col = "KMeans_Cluster" if winner == "K-Means" else "Hier_Cluster"
print(f"\nWinner (larger prevalence spread): {winner}")

# --- PCA visualization of the winning clustering ---
plt.figure(figsize=(7, 6))
scatter = plt.scatter(df["PCA1"], df["PCA2"], c=df[winning_col], cmap="viridis", alpha=0.6, s=25)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title(f"Patient Clusters ({winner}, k={best_k}) — PCA Projection")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/clusters_pca.png", dpi=150)
plt.close()

# --- Cluster profiles + prevalence bar chart for the winner ---
profile = df.groupby(winning_col)[["age", "sysBP", "totChol", "BMI", "glucose"]].mean().round(1)
prevalence = df.groupby(winning_col)["TenYearCHD"].mean().round(3) * 100
profile["CHD_Prevalence_%"] = prevalence
profile["N_Patients"] = df.groupby(winning_col).size()
profile.to_csv("results/cluster_profiles.csv")

plt.figure(figsize=(6, 4.5))
prevalence.plot(kind="bar", color="#C44E52")
plt.ylabel("10-Year CHD Prevalence (%)")
plt.xlabel("Cluster")
plt.title(f"CHD Prevalence by Cluster ({winner})")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cluster_chd_prevalence.png", dpi=150)
plt.close()

overall_prevalence = df["TenYearCHD"].mean() * 100
spread = max(km_spread, hier_spread)
is_fruitful = spread >= 10  # meaningful separation bar for this harder, imbalanced dataset

with open("results/clustering_notes.json", "w") as f:
    json.dump({
        "best_k": int(best_k),
        "kmeans_silhouette": round(max(sil_scores_km), 4),
        "hierarchical_silhouette": round(sil_hier, 4),
        "kmeans_prevalence_spread_pp": round(km_spread, 1),
        "hierarchical_prevalence_spread_pp": round(hier_spread, 1),
        "winner": winner,
        "overall_chd_prevalence_%": round(overall_prevalence, 1),
        "kept": bool(is_fruitful),
    }, f, indent=2)

print(f"\nOverall CHD prevalence: {overall_prevalence:.1f}%")
print(f"Clustering kept: {is_fruitful}")
