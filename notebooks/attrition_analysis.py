# %% [markdown]
# # Employee Attrition Prediction
# Predicting which employees are at risk of leaving, and identifying the key
# drivers of attrition, using the IBM HR Analytics Employee Attrition dataset.
#
# Dataset source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Download the CSV (WA_Fn-UseC_-HR-Employee-Attrition.csv), place it in the
# `data/` folder, and rename it to `attrition.csv` before running this script.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not installed — run: pip install imbalanced-learn")

sns.set_style("whitegrid")
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load data

# %%
df = pd.read_csv("../data/attrition.csv")
print(df.shape)
df.head()

# %% [markdown]
# ## 2. Clean & prepare
# Drop constant/non-informative columns, encode target, engineer features.

# %%
drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Feature engineering
df["TenureBucket"] = pd.cut(
    df["YearsAtCompany"], bins=[-1, 2, 5, 10, 100],
    labels=["0-2yrs", "3-5yrs", "6-10yrs", "10+yrs"]
)
df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"]

print("Attrition rate: {:.1f}%".format(df["Attrition"].mean() * 100))

# %% [markdown]
# ## 3. Exploratory data analysis

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.barplot(data=df, x="OverTime", y="Attrition", ax=axes[0])
axes[0].set_title("Attrition Rate by Overtime")

sns.barplot(data=df, x="TenureBucket", y="Attrition", ax=axes[1])
axes[1].set_title("Attrition Rate by Tenure")
axes[1].tick_params(axis="x", rotation=30)

sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=axes[2])
axes[2].set_title("Monthly Income vs Attrition")

plt.tight_layout()
plt.savefig("../outputs/eda_summary.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Preprocessing — encode categoricals, split, scale

# %%
target = "Attrition"
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols += [c for c in ["TenureBucket"] if c in df.columns]

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 5. Handle class imbalance with SMOTE

# %%
if HAS_SMOTE:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
    print("Before SMOTE:", y_train.value_counts().to_dict())
    print("After SMOTE:", pd.Series(y_train_bal).value_counts().to_dict())
else:
    X_train_bal, y_train_bal = X_train_scaled, y_train

# %% [markdown]
# ## 6. Train models

# %%
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_bal, y_train_bal)

rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
rf.fit(X_train_bal, y_train_bal)

# %% [markdown]
# ## 7. Evaluate — precision/recall/F1/ROC-AUC, not just accuracy

# %%
def evaluate(model, name):
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    print(f"--- {name} ---")
    print(classification_report(y_test, preds, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, probs), 3))
    print()
    return probs

lr_probs = evaluate(log_reg, "Logistic Regression")
rf_probs = evaluate(rf, "Random Forest")

# %%
fig, ax = plt.subplots(figsize=(6, 5))
for probs, label in [(lr_probs, "Logistic Regression"), (rf_probs, "Random Forest")]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, probs):.2f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.savefig("../outputs/roc_curve.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Feature importance — the "why", not just the "who"

# %%
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index, orient="h")
plt.title("Top 10 Attrition Drivers (Random Forest Feature Importance)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("../outputs/feature_importance.png", dpi=150)
plt.show()

print(top_features)

# %% [markdown]
# ## 9. Export risk scores for Power BI
# Score the full dataset and export a CSV with predicted attrition risk per
# employee — this feeds the "at-risk employee" Power BI dashboard.

# %%
X_full_scaled = scaler.transform(X)
risk_scores = rf.predict_proba(X_full_scaled)[:, 1]

export_df = df.copy()
export_df["AttritionRiskScore"] = risk_scores
export_df["RiskTier"] = pd.cut(
    export_df["AttritionRiskScore"],
    bins=[-0.01, 0.33, 0.66, 1],
    labels=["Low", "Medium", "High"]
)

export_df.to_csv("../outputs/attrition_risk_scores.csv", index=False)
print("Exported: outputs/attrition_risk_scores.csv")
print(export_df[["RiskTier"]].value_counts())
