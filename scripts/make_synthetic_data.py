"""
Generates a synthetic dataset with the same column schema as the IBM HR
Analytics Employee Attrition dataset, purely so the pipeline can be test-run
without network access to Kaggle. This is NOT a substitute for the real
dataset — download the real one from Kaggle for your actual project:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1470

departments = ["Sales", "Research & Development", "Human Resources"]
job_roles = ["Sales Executive", "Research Scientist", "Laboratory Technician",
             "Manufacturing Director", "Healthcare Representative", "Manager",
             "Sales Representative", "Research Director", "Human Resources"]
education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                     "Human Resources", "Other"]
marital_status = ["Single", "Married", "Divorced"]
business_travel = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]

overtime = rng.choice(["Yes", "No"], size=n, p=[0.28, 0.72])
years_at_company = rng.integers(0, 25, size=n)
years_since_promo = np.clip(years_at_company - rng.integers(0, 10, size=n), 0, None)
work_life_balance = rng.integers(1, 5, size=n)
job_satisfaction = rng.integers(1, 5, size=n)
monthly_income = rng.integers(1000, 20000, size=n)

# attrition probability driven loosely by overtime, tenure, satisfaction, income
risk = (
    0.35 * (overtime == "Yes").astype(float)
    + 0.20 * (years_since_promo > 4).astype(float)
    + 0.20 * (work_life_balance <= 2).astype(float)
    + 0.15 * (job_satisfaction <= 2).astype(float)
    + 0.10 * (monthly_income < 4000).astype(float)
)
attrition_prob = np.clip(risk * 0.4, 0.02, 0.75)
attrition = rng.binomial(1, attrition_prob)

df = pd.DataFrame({
    "Age": rng.integers(18, 60, size=n),
    "Attrition": np.where(attrition == 1, "Yes", "No"),
    "BusinessTravel": rng.choice(business_travel, size=n, p=[0.6, 0.2, 0.2]),
    "DailyRate": rng.integers(100, 1500, size=n),
    "Department": rng.choice(departments, size=n, p=[0.3, 0.6, 0.1]),
    "DistanceFromHome": rng.integers(1, 30, size=n),
    "Education": rng.integers(1, 5, size=n),
    "EducationField": rng.choice(education_fields, size=n),
    "EmployeeCount": 1,
    "EmployeeNumber": np.arange(1, n + 1),
    "EnvironmentSatisfaction": rng.integers(1, 5, size=n),
    "Gender": rng.choice(["Male", "Female"], size=n),
    "HourlyRate": rng.integers(30, 100, size=n),
    "JobInvolvement": rng.integers(1, 5, size=n),
    "JobLevel": rng.integers(1, 5, size=n),
    "JobRole": rng.choice(job_roles, size=n),
    "JobSatisfaction": job_satisfaction,
    "MaritalStatus": rng.choice(marital_status, size=n),
    "MonthlyIncome": monthly_income,
    "MonthlyRate": rng.integers(2000, 27000, size=n),
    "NumCompaniesWorked": rng.integers(0, 9, size=n),
    "Over18": "Y",
    "OverTime": overtime,
    "PercentSalaryHike": rng.integers(11, 25, size=n),
    "PerformanceRating": rng.choice([3, 4], size=n, p=[0.85, 0.15]),
    "RelationshipSatisfaction": rng.integers(1, 5, size=n),
    "StandardHours": 80,
    "StockOptionLevel": rng.integers(0, 4, size=n),
    "TotalWorkingYears": years_at_company + rng.integers(0, 5, size=n),
    "TrainingTimesLastYear": rng.integers(0, 6, size=n),
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": np.minimum(years_at_company, rng.integers(0, 15, size=n)),
    "YearsSinceLastPromotion": years_since_promo,
    "YearsWithCurrManager": np.minimum(years_at_company, rng.integers(0, 15, size=n)),
})

df.to_csv("../data/attrition.csv", index=False)
print("Synthetic dataset written to data/attrition.csv:", df.shape)
print("Attrition rate:", (df["Attrition"] == "Yes").mean())
