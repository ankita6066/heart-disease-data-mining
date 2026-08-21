"""Exploratory Data Analysis for the Framingham Heart Study dataset."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIG_DIR = "results/figures"
df = pd.read_csv("data/framingham_clean.csv")

NUMERIC_COLS = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

# 1. Class balance (this dataset is notably imbalanced ~85/15)
plt.figure(figsize=(5, 4))
counts = df["TenYearCHD"].value_counts().sort_index()
plt.bar(["No CHD (0)", "CHD in 10yr (1)"], counts.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(counts.values):
    plt.text(i, v + 20, f"{v}\n({v/len(df)*100:.1f}%)", ha="center")
plt.title("Class Distribution (Imbalanced)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=150)
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
corr_cols = NUMERIC_COLS + ["male", "currentSmoker", "BPMeds", "prevalentStroke",
                            "prevalentHyp", "diabetes", "TenYearCHD"]
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, annot_kws={"size": 7})
plt.title("Feature Correlation Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# 3. Numeric feature distributions by outcome
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, feat in zip(axes.ravel(), NUMERIC_COLS):
    sns.kdeplot(data=df, x=feat, hue="TenYearCHD", fill=True, alpha=0.4, ax=ax, legend=False)
    ax.set_title(feat)
fig.suptitle("Feature Distributions by Outcome (0=No CHD, 1=CHD)", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Risk by age group and sex
plt.figure(figsize=(8, 5))
risk_by_age_sex = df.groupby(["Age_Group", "male"], observed=True)["TenYearCHD"].mean().unstack() * 100
risk_by_age_sex.columns = ["Female", "Male"]
risk_by_age_sex.plot(kind="bar", ax=plt.gca(), color=["#DD8452", "#4C72B0"])
plt.ylabel("10-Year CHD Risk (%)")
plt.title("CHD Risk by Age Group and Sex")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/risk_by_age_sex.png", dpi=150)
plt.close()

print("EDA complete. Correlation with TenYearCHD:")
print(corr["TenYearCHD"].sort_values(ascending=False))
print(f"\nClass imbalance: {counts[1]} positive / {len(df)} total = {counts[1]/len(df)*100:.1f}%")
