"""
Generates a static dashboard-style PNG from outputs/attrition_risk_scores.csv,
for embedding in the GitHub README. This is a placeholder built from pipeline
output — replace with a real Power BI screenshot once you build the .pbix on
the real Kaggle dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "DejaVu Sans"

df = pd.read_csv("../outputs/attrition_risk_scores.csv")
df = df.reset_index().rename(columns={"index": "EmployeeID"})
df["EmployeeID"] = df["EmployeeID"] + 1

# ---- derived KPIs ----
total_employees = len(df)
attrition_rate = (df["Attrition"] == "Yes").mean() * 100
high_risk_count = (df["RiskTier"] == "High").sum()
avg_risk_score = df["AttritionRiskScore"].mean() * 100

navy = "#0B2545"
accent = "#1F6FEB"
red = "#D9534F"
amber = "#E8A33D"
green = "#3FA34D"
gray = "#6B7280"
bg = "#F7F9FC"

fig = plt.figure(figsize=(14, 8), facecolor=bg)
gs = gridspec.GridSpec(
    3, 4, figure=fig, height_ratios=[0.55, 1.5, 1.5],
    hspace=0.55, wspace=0.4, left=0.05, right=0.97, top=0.90, bottom=0.06
)

# ---- Title ----
fig.text(0.05, 0.955, "Employee Attrition Risk Dashboard", fontsize=20,
          fontweight="bold", color=navy)
fig.text(0.05, 0.925, "HR Workforce Analytics  |  Model: Random Forest Classifier  |  Data: IBM HR Analytics (sample run)",
          fontsize=10, color=gray)

# ---- KPI cards ----
kpis = [
    ("Total Employees", f"{total_employees:,}", navy),
    ("Overall Attrition Rate", f"{attrition_rate:.1f}%", red),
    ("High-Risk Employees", f"{high_risk_count:,}", amber),
    ("Avg. Predicted Risk", f"{avg_risk_score:.1f}%", accent),
]
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor("white")
    ax.axis("off")
    box = mpatches.FancyBboxPatch((0.03, 0.05), 0.94, 0.9,
                                   boxstyle="round,pad=0.02,rounding_size=0.06",
                                   transform=ax.transAxes,
                                   facecolor="white", edgecolor="#E5E9F0", linewidth=1)
    ax.add_patch(box)
    ax.text(0.5, 0.62, value, transform=ax.transAxes, ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0.5, 0.22, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=gray)

# ---- Risk tier donut ----
ax1 = fig.add_subplot(gs[1, 0])
tier_counts = df["RiskTier"].value_counts().reindex(["Low", "Medium", "High"])
colors = [green, amber, red]
wedges, _ = ax1.pie(tier_counts, colors=colors, startangle=90,
                     wedgeprops=dict(width=0.42, edgecolor="white"))
ax1.text(0, 0, f"{total_employees:,}\nEmployees", ha="center", va="center",
         fontsize=11, fontweight="bold", color=navy)
ax1.set_title("Risk Tier Distribution", fontsize=12, fontweight="bold", color=navy, pad=10)
ax1.legend(tier_counts.index, loc="lower center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, frameon=False, fontsize=9)

# ---- Attrition rate by department ----
ax2 = fig.add_subplot(gs[1, 1:3])
dept_rate = (df.assign(is_attr=(df["Attrition"] == "Yes").astype(int))
             .groupby("Department")["is_attr"].mean().sort_values(ascending=False) * 100)
bars = ax2.barh(dept_rate.index, dept_rate.values, color=accent, height=0.5)
ax2.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9, color=navy)
ax2.set_title("Attrition Rate by Department", fontsize=12, fontweight="bold", color=navy, pad=10)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(labelsize=9)
ax2.set_xlim(0, max(dept_rate.values.max() * 1.3, 1))

# ---- Overtime comparison ----
ax3 = fig.add_subplot(gs[1, 3])
ot_rate = (df.assign(is_attr=(df["Attrition"] == "Yes").astype(int))
           .groupby("OverTime")["is_attr"].mean() * 100).reindex(["No", "Yes"])
bars = ax3.bar(ot_rate.index, ot_rate.values, color=[green, red], width=0.5)
ax3.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9, color=navy)
ax3.set_title("Attrition by Overtime", fontsize=12, fontweight="bold", color=navy, pad=10)
ax3.spines[["top", "right"]].set_visible(False)
ax3.tick_params(labelsize=9)
ax3.set_ylim(0, max(max(ot_rate.values) * 1.4, 1))

# ---- Top attrition drivers (static illustrative ranking) ----
ax4 = fig.add_subplot(gs[2, 0:2])
drivers = pd.Series({
    "OverTime": 7.2, "Education": 5.6, "WorkLifeBalance": 5.4,
    "RelationshipSatisfaction": 3.7, "MonthlyRate": 3.5, "JobSatisfaction": 3.0,
}).sort_values()
bars = ax4.barh(drivers.index, drivers.values, color=navy, height=0.55)
ax4.set_title("Top Attrition Drivers (Feature Importance)", fontsize=12,
              fontweight="bold", color=navy, pad=10)
ax4.spines[["top", "right"]].set_visible(False)
ax4.tick_params(labelsize=9)
ax4.set_xlabel("Relative Importance (%)", fontsize=9, color=gray)

# ---- High-risk watchlist table ----
ax5 = fig.add_subplot(gs[2, 2:4])
ax5.axis("off")
ax5.set_title("At-Risk Employee Watchlist (Top 6)", fontsize=12,
              fontweight="bold", color=navy, pad=10, loc="left")
watch_cols = ["EmployeeID", "Department", "JobRole", "AttritionRiskScore"]
watchlist = (df.sort_values("AttritionRiskScore", ascending=False)
             .head(6)[watch_cols].copy())
watchlist["AttritionRiskScore"] = (watchlist["AttritionRiskScore"] * 100).round(1).astype(str) + "%"
watchlist.columns = ["Emp #", "Department", "Job Role", "Risk"]

table = ax5.table(cellText=watchlist.values, colLabels=watchlist.columns,
                   loc="center", cellLoc="left", colLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#E5E9F0")
    if row == 0:
        cell.set_facecolor(navy)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("white" if row % 2 else "#F3F6FB")

fig.text(0.05, 0.015,
         "Placeholder dashboard generated from pipeline output — replace with a Power BI screenshot built on the real Kaggle dataset.",
         fontsize=8, color=gray, style="italic")

plt.savefig("../outputs/dashboard_preview.png", dpi=160, facecolor=bg, bbox_inches="tight")
print("Saved outputs/dashboard_preview.png")
