"""
Preprocessing for the Framingham Heart Study dataset.

Unlike the Pima diabetes dataset (which encoded missingness as 0), this
dataset has genuine NaN values across 7 columns. We use median/mode
imputation grouped by outcome where sensible, then engineer a few
clinically meaningful categorical features for later association-rule
mining and interpretability.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
BINARY_COLS = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + ["education"]
TARGET_COL = "TenYearCHD"


def load_raw(path="data/framingham.csv"):
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # education is an ordinal category with missing values -> impute with mode
    df["education"] = df["education"].fillna(df["education"].mode()[0])

    # BPMeds is binary (on blood pressure medication) -> missing almost certainly means "no"
    df["BPMeds"] = df["BPMeds"].fillna(0)

    # Continuous clinical measurements -> median imputation within outcome class,
    # so imputed values reflect a realistic CHD-risk vs. no-CHD-risk profile
    for col in ["cigsPerDay", "totChol", "BMI", "heartRate", "glucose"]:
        df[col] = df.groupby(TARGET_COL)[col].transform(lambda s: s.fillna(s.median()))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age_Group"] = pd.cut(df["age"], bins=[0, 40, 50, 60, 100],
                              labels=["Under40", "40s", "50s", "60Plus"])
    df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                                 labels=["Underweight", "Normal", "Overweight", "Obese"])
    df["BP_Category"] = pd.cut(df["sysBP"], bins=[0, 120, 140, 300],
                                labels=["Normal", "Elevated", "High"])
    df["Chol_Category"] = pd.cut(df["totChol"], bins=[0, 200, 240, 700],
                                  labels=["Desirable", "Borderline", "High"])
    df["Smoker_Category"] = np.where(df["currentSmoker"] == 1, "Smoker", "NonSmoker")
    return df


def get_train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_raw()
    print(f"Raw shape: {df.shape}, missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    df = clean(df)
    df = engineer_features(df)
    df.to_csv("data/framingham_clean.csv", index=False)
    print(f"\nCleaned shape: {df.shape}")
    print(f"Missing values after cleaning: {df.isna().sum().sum()}")
    print(df.head())
