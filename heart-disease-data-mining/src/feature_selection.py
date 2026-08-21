"""
Feature selection using two independent methods:
1. Mutual information (captures non-linear relationships with the target)
2. Recursive Feature Elimination with a Random Forest (captures interaction effects)

We keep a feature only if it's flagged as important by at least one method,
and report where the two methods agree vs. disagree.
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, "src")
from preprocessing import load_raw, clean, engineer_features, get_train_test_split, FEATURE_COLS

FIG_DIR = "results/figures"
df = engineer_features(clean(load_raw()))
X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = get_train_test_split(df)

# --- Method 1: Mutual Information ---
mi_scores = mutual_info_classif(X_train_s, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=FEATURE_COLS).sort_values(ascending=False)

# --- Method 2: Recursive Feature Elimination (RFE) with Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=8)
rfe.fit(X_train_s, y_train)
rfe_selected = set(str(f) for f in np.array(FEATURE_COLS)[rfe.support_])

# --- Combine: keep top-8 MI features union RFE-selected features ---
mi_top8 = set(mi_series.head(8).index)
consensus_features = sorted(mi_top8 & rfe_selected)
union_features = sorted(mi_top8 | rfe_selected)

print("Top features by Mutual Information:")
print(mi_series.head(10))
print(f"\nRFE-selected features: {sorted(rfe_selected)}")
print(f"\nConsensus (both methods agree): {consensus_features}")
print(f"Union (either method flags): {union_features}")

# Plot MI scores
plt.figure(figsize=(8, 6))
mi_series.sort_values().plot(kind="barh", color="#55A868")
plt.title("Feature Importance by Mutual Information")
plt.xlabel("Mutual Information Score")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_selection_mi.png", dpi=150)
plt.close()

with open("results/feature_selection_notes.json", "w") as f:
    json.dump({
        "mi_ranking": mi_series.round(4).to_dict(),
        "rfe_selected": sorted(rfe_selected),
        "consensus_features": consensus_features,
        "recommended_feature_set": union_features,
    }, f, indent=2)

print(f"\nRecommended feature set for modeling ({len(union_features)} features):", union_features)
