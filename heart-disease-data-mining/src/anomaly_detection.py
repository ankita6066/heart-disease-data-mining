"""
Anomaly detection using Isolation Forest to flag clinically atypical patients
-- individuals whose combination of vitals is unusual relative to the rest
of the cohort. We then check post-hoc whether flagged patients have
elevated CHD risk, which would suggest the anomaly score itself carries
clinical signal (unusual vitals profiles co-occurring with elevated risk).
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

import sys
sys.path.insert(0, "src")
from preprocessing import load_raw, clean, engineer_features, FEATURE_COLS

FIG_DIR = "results/figures"
df = engineer_features(clean(load_raw()))
X = df[FEATURE_COLS]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
df["Anomaly"] = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
df["Anomaly_Score"] = iso.decision_function(X_scaled)  # lower = more anomalous

n_anomalies = (df["Anomaly"] == -1).sum()
print(f"Flagged {n_anomalies} anomalous patients ({n_anomalies/len(df)*100:.1f}% of cohort)")

# --- Statistical test: do anomalous patients have significantly different CHD risk? ---
normal_risk = df[df["Anomaly"] == 1]["TenYearCHD"]
anomaly_risk = df[df["Anomaly"] == -1]["TenYearCHD"]
anomaly_prevalence = anomaly_risk.mean() * 100
normal_prevalence = normal_risk.mean() * 100

# Two-proportion z-test via chi-square (contingency table)
contingency = pd.crosstab(df["Anomaly"], df["TenYearCHD"])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"CHD prevalence — anomalous patients: {anomaly_prevalence:.1f}%, normal patients: {normal_prevalence:.1f}%")
print(f"Chi-square test: chi2={chi2:.2f}, p-value={p_value:.6f}")

is_fruitful = p_value < 0.01 and abs(anomaly_prevalence - normal_prevalence) >= 5

# --- Visualization: anomalies on PCA projection ---
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
plt.figure(figsize=(7, 6))
normal_mask = df["Anomaly"] == 1
plt.scatter(coords[normal_mask, 0], coords[normal_mask, 1], c="#4C72B0", alpha=0.4, s=15, label="Normal")
plt.scatter(coords[~normal_mask, 0], coords[~normal_mask, 1], c="#C44E52", alpha=0.8, s=35,
            marker="x", label="Anomalous")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("Isolation Forest — Anomalous Patients Flagged")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/anomaly_detection_pca.png", dpi=150)
plt.close()

# --- Bar chart: CHD prevalence, normal vs anomalous ---
plt.figure(figsize=(5, 4.5))
plt.bar(["Normal", "Anomalous"], [normal_prevalence, anomaly_prevalence], color=["#4C72B0", "#C44E52"])
plt.ylabel("10-Year CHD Prevalence (%)")
plt.title("CHD Risk: Normal vs. Anomalous Patients")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/anomaly_chd_prevalence.png", dpi=150)
plt.close()

with open("results/anomaly_detection_notes.json", "w") as f:
    json.dump({
        "n_flagged": int(n_anomalies),
        "pct_flagged": round(n_anomalies / len(df) * 100, 1),
        "anomaly_chd_prevalence_%": round(anomaly_prevalence, 1),
        "normal_chd_prevalence_%": round(normal_prevalence, 1),
        "chi2_statistic": round(chi2, 2),
        "p_value": p_value,
        "kept": bool(is_fruitful),
    }, f, indent=2)

print(f"\nAnomaly detection result kept: {is_fruitful}")
