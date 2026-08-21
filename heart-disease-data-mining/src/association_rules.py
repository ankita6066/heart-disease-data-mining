"""
Association rule mining (Apriori) on discretized clinical + lifestyle
features to surface interpretable if-then patterns linked to 10-year CHD risk.
"""
import pandas as pd
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import sys
sys.path.insert(0, "src")
from preprocessing import load_raw, clean, engineer_features

df = engineer_features(clean(load_raw()))
df["Outcome_Label"] = df["TenYearCHD"].apply(lambda x: "CHD" if x == 1 else "No_CHD")
df["Hypertension"] = df["prevalentHyp"].apply(lambda x: "Hypertensive" if x == 1 else "Normotensive")
df["Diabetes_Status"] = df["diabetes"].apply(lambda x: "Diabetic" if x == 1 else "Non_Diabetic")
df["Sex"] = df["male"].apply(lambda x: "Male" if x == 1 else "Female")

cat_cols = ["Age_Group", "BMI_Category", "BP_Category", "Chol_Category",
            "Smoker_Category", "Hypertension", "Diabetes_Status", "Sex", "Outcome_Label"]
transactions = df[cat_cols].astype(str).values.tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
print(f"Frequent itemsets found: {len(frequent_itemsets)}")

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

chd_rules = rules[rules["consequents"].apply(lambda x: "CHD" in x or "No_CHD" in x)]

# Keep only rules that: predict CHD status specifically (not No_CHD, which is
# trivially easy to predict given the 85% base rate), have decent confidence,
# and a real positive lift.
fruitful_rules = chd_rules[
    chd_rules["consequents"].apply(lambda x: "CHD" in x and "No_CHD" not in x) &
    (chd_rules["confidence"] >= 0.25) &
    (chd_rules["lift"] >= 1.3) &
    (chd_rules["support"] >= 0.02)
].sort_values("lift", ascending=False)

fruitful_display = fruitful_rules.copy()
fruitful_display["antecedents"] = fruitful_display["antecedents"].apply(lambda x: ", ".join(sorted(x)))
fruitful_display["consequents"] = fruitful_display["consequents"].apply(lambda x: ", ".join(sorted(x)))
fruitful_display = fruitful_display[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)

fruitful_display.to_csv("results/association_rules.csv", index=False)
print(f"\nFruitful CHD-predicting rules kept: {len(fruitful_display)} (out of {len(chd_rules)} CHD-related rules found)")
print(fruitful_display.head(15).to_string(index=False))

is_fruitful = len(fruitful_display) > 0
with open("results/association_rules_notes.json", "w") as f:
    json.dump({
        "total_frequent_itemsets": len(frequent_itemsets),
        "total_chd_related_rules": len(chd_rules),
        "fruitful_rules_kept": len(fruitful_display),
        "kept": bool(is_fruitful),
    }, f, indent=2)
