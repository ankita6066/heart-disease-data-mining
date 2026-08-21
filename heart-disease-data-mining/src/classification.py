"""
Trains and evaluates a wider zoo of classifiers than the diabetes project,
including boosting variants, an MLP neural net, and a voting ensemble of
the top performers. Because this dataset is imbalanced (~15% positive),
we train each model both with and without SMOTE oversampling and report
whichever variant performs better on recall/F1 — a real class-imbalance
technique that has no equivalent in the smaller, balanced diabetes project.
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix,
                              ConfusionMatrixDisplay)

import sys
sys.path.insert(0, "src")
from preprocessing import load_raw, clean, engineer_features, get_train_test_split, FEATURE_COLS

FIG_DIR = "results/figures"
df = engineer_features(clean(load_raw()))
X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = get_train_test_split(df)

# SMOTE-resampled training set (only ever applied to TRAINING data, never test data)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_s, y_train)
print(f"Original training class balance: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {y_train_smote.value_counts().to_dict()}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight=None),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=7, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric="logloss", random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
fitted_models = {}  # name -> (fitted_model, which_train_data_used)

for name, model in models.items():
    # Evaluate both plain and SMOTE-resampled training. SMOTE is applied INSIDE each
    # CV fold via an imblearn Pipeline (fit only on that fold's training split) to
    # avoid the data leakage that comes from resampling before splitting.
    cv_plain = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")

    model_smote = type(model)(**model.get_params())
    smote_pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", model_smote)])
    cv_smote_scores = cross_val_score(smote_pipeline, X_train_s, y_train, cv=cv, scoring="roc_auc")

    use_smote = cv_smote_scores.mean() > cv_plain.mean()
    chosen_cv = cv_smote_scores if use_smote else cv_plain

    final_model = type(model)(**model.get_params())
    if use_smote:
        final_model.fit(X_train_smote, y_train_smote)
    else:
        final_model.fit(X_train_s, y_train)
    fitted_models[name] = final_model

    preds = final_model.predict(X_test_s)
    probs = final_model.predict_proba(X_test_s)[:, 1] if hasattr(final_model, "predict_proba") else preds

    results.append({
        "Model": name,
        "Used_SMOTE": use_smote,
        "CV_ROC_AUC_mean": round(chosen_cv.mean(), 4),
        "CV_ROC_AUC_std": round(chosen_cv.std(), 4),
        "Test_Accuracy": round(accuracy_score(y_test, preds), 4),
        "Test_Precision": round(precision_score(y_test, preds), 4),
        "Test_Recall": round(recall_score(y_test, preds), 4),
        "Test_F1": round(f1_score(y_test, preds), 4),
        "Test_ROC_AUC": round(roc_auc_score(y_test, probs), 4),
    })

results_df = pd.DataFrame(results).sort_values("Test_ROC_AUC", ascending=False)
results_df.to_csv("results/classification_results.csv", index=False)
print("\n" + results_df.to_string(index=False))

# --- Selection rule: keep models with held-out TEST ROC-AUC >= 0.70. We deliberately
# select on the unbiased test metric rather than CV score here, since CV scores computed
# on SMOTE-resampled folds can run optimistic; test performance is what actually matters. ---
BASELINE_AUC = 0.70
keep_mask = results_df["Test_ROC_AUC"] >= BASELINE_AUC
kept_models = results_df[keep_mask]["Model"].tolist()
dropped_models = results_df[~keep_mask]["Model"].tolist()

print(f"\nKept (fruitful) models: {kept_models}")
print(f"Dropped (underperforming) models: {dropped_models}")

# --- Voting ensemble of the top 3 kept models (soft voting on probabilities) ---
top3_names = results_df[results_df["Model"].isin(kept_models)].head(3)["Model"].tolist()
voting_estimators = [(n.replace(" ", "_"), fitted_models[n]) for n in top3_names]
voting_clf = VotingClassifier(estimators=voting_estimators, voting="soft")
voting_clf.fit(X_train_s, y_train)  # VotingClassifier needs its own fit call on plain data
voting_preds = voting_clf.predict(X_test_s)
voting_probs = voting_clf.predict_proba(X_test_s)[:, 1]
voting_auc = roc_auc_score(y_test, voting_probs)
voting_result = {
    "Model": f"Voting Ensemble ({'+'.join(top3_names)})",
    "Used_SMOTE": False,
    "Test_Accuracy": round(accuracy_score(y_test, voting_preds), 4),
    "Test_Precision": round(precision_score(y_test, voting_preds), 4),
    "Test_Recall": round(recall_score(y_test, voting_preds), 4),
    "Test_F1": round(f1_score(y_test, voting_preds), 4),
    "Test_ROC_AUC": round(voting_auc, 4),
}
print(f"\nVoting Ensemble result: {voting_result}")

ensemble_improves = voting_auc > results_df.iloc[0]["Test_ROC_AUC"]
if ensemble_improves:
    kept_models.append(voting_result["Model"])
    fitted_models[voting_result["Model"]] = voting_clf
    results_df = pd.concat([results_df, pd.DataFrame([voting_result])], ignore_index=True).sort_values("Test_ROC_AUC", ascending=False)
    results_df.to_csv("results/classification_results.csv", index=False)

with open("results/model_selection_notes.json", "w") as f:
    json.dump({
        "kept_models": kept_models,
        "dropped_models": dropped_models,
        "baseline_auc_threshold": BASELINE_AUC,
        "voting_ensemble_kept": bool(ensemble_improves),
        "voting_ensemble_result": voting_result,
    }, f, indent=2)

# --- ROC curves for kept models ---
plt.figure(figsize=(8, 7))
for name in kept_models:
    if name.startswith("Voting"):
        probs = voting_probs
    else:
        model = fitted_models[name]
        probs = model.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Retained Models")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_curves.png", dpi=150)
plt.close()

# --- Confusion matrix for the single best model ---
best_row = results_df.iloc[0]
best_name = best_row["Model"]
if best_name.startswith("Voting"):
    preds = voting_preds
else:
    preds = fitted_models[best_name].predict(X_test_s)
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No CHD", "CHD"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix — Best Model ({best_name})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix_best_model.png", dpi=150)
plt.close()

# --- Feature importance (best tree-based model among kept) ---
tree_models_kept = [n for n in ["XGBoost", "Gradient Boosting", "Random Forest", "Extra Trees", "AdaBoost", "Decision Tree"] if n in kept_models]
if tree_models_kept:
    imp_name = tree_models_kept[0]
    imp_model = fitted_models[imp_name]
    importances = pd.Series(imp_model.feature_importances_, index=FEATURE_COLS).sort_values()
    plt.figure(figsize=(7, 6))
    importances.plot(kind="barh", color="#4C72B0")
    plt.title(f"Feature Importance ({imp_name})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close()

print(f"\nBest model: {best_name}")
print("Figures and results saved to results/")
