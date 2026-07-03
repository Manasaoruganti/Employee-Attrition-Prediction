# Employee Attrition Prediction

Predicting which employees are at risk of leaving, and identifying the key
factors driving attrition, so HR can intervene proactively rather than
reactively.

## Business Problem
Employee attrition is costly — every departure creates recruitment costs,
lost productivity, and knowledge loss. HR teams often react to attrition
*after* it happens. **Can we predict which employees are at high risk of
leaving, and identify the key factors driving attrition, so HR can act
before it happens?**

## Dataset
[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
(Kaggle, public) — 1,470 employee records, 35 features covering demographics,
compensation, satisfaction, tenure, and performance. A fictional but
realistic dataset created by IBM data scientists, widely used as an HR
analytics benchmark.

## Tools
- **Python** (Pandas, NumPy) — data cleaning & feature engineering
- **Matplotlib / Seaborn** — exploratory data analysis
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, evaluation
- **imbalanced-learn (SMOTE)** — correcting class imbalance
- **Power BI** — business-facing "at-risk employee" dashboard

## Methodology
1. Cleaned the dataset and engineered features (tenure buckets, promotion
   gap ratio, income per job level)
2. Explored attrition patterns across overtime, tenure, and income
3. Encoded categoricals, scaled numeric features, split train/test
4. Applied **SMOTE** to correct class imbalance (attrition is a minority class)
5. Trained **Logistic Regression** (interpretable baseline) and
   **Random Forest** (stronger performance, feature importance)
6. Evaluated with **precision, recall, F1-score, and ROC-AUC** — not just
   accuracy, since accuracy is misleading on imbalanced data
7. Extracted feature importance to identify the top attrition drivers
8. Exported per-employee risk scores to `outputs/attrition_risk_scores.csv`
   for the Power BI dashboard

## Dashboard Preview

![Employee Attrition Risk Dashboard](outputs/dashboard_preview.png)
<img width="2349" height="1274" alt="dashboard_preview" src="https://github.com/user-attachments/assets/5a93377e-2639-445d-b06f-a261bc1241da" />


> **Note:** This preview is generated directly from the pipeline's output
> (`outputs/attrition_risk_scores.csv`) so the repo shows a visual result
> without requiring Power BI to view it. Replace this image with a screenshot
> of your actual `.pbix` dashboard once you build it on the real Kaggle
> dataset — see `powerbi/` and the "How to Run" section below.

## Key Findings
*(Fill in with your own numbers once you run this on the real Kaggle dataset
— the values below are placeholders from a test run on synthetic data.)*
- Overtime, tenure-related factors, and satisfaction scores are consistently
  among the top predictors of attrition
- The Random Forest model achieves an ROC-AUC of ~0.7, showing meaningful
  (though not perfect) predictive power given the dataset's size
- Class imbalance (most employees stay) means precision/recall trade-offs
  matter more than raw accuracy

## Business Recommendation
HR should prioritize retention conversations for employee segments showing
the top 2-3 risk factors identified by the model (e.g., frequent overtime
combined with a long gap since last promotion), rather than treating
attrition as unpredictable.

## Repository Structure
```
employee-attrition-prediction/
├── data/                          # place the Kaggle CSV here (not committed)
├── notebooks/
│   └── attrition_analysis.py      # full pipeline (jupytext "percent" format)
├── outputs/                       # generated charts + risk-scored CSV
├── powerbi/                       # Power BI dashboard file
├── scripts/
│   ├── make_synthetic_data.py     # optional: generate test data if offline
│   └── make_dashboard_preview.py  # generates outputs/dashboard_preview.png
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt

# 1. Download the real dataset from Kaggle (see link above), rename to
#    attrition.csv, and place it in data/
#    (or, to test the pipeline without Kaggle access, run:)
cd scripts && python make_synthetic_data.py && cd ..

# 2. Run the analysis (as a script)
cd notebooks && python attrition_analysis.py && cd ..

# — or open it as a notebook —
pip install jupytext
cd notebooks && jupytext --to notebook attrition_analysis.py && jupyter notebook attrition_analysis.ipynb && cd ..

# 3. (Optional) Regenerate the README dashboard preview image
cd scripts && python make_dashboard_preview.py && cd ..
```

## Author
Manasa Oruganti — [LinkedIn](https://www.linkedin.com/in/orugantimanasa/) · [GitHub](https://github.com/Manasaoruganti)
