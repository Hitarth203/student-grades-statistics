#Combining all of them and writing her using Chatgpt (Copy-paste)

# student_performance_analysis.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# ---------- SETUP ----------
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/student-mat.csv", sep=';')
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# ---------- PHASE 1: DESCRIPTIVE STATISTICS ----------
features = ['age', 'absences', 'studytime', 'failures', 'G1', 'G2', 'G3']

print("\n📊 Summary Statistics:")
print(df[features].agg(['mean', 'median', 'std', 'min', 'max']).round(2))

print("\n📐 Skewness and Kurtosis:")
print(df[features].agg(['skew', 'kurt']).T.round(2))

# ---------- PHASE 2: DISTRIBUTIONS & PROBABILITY ----------

# Histogram + KDE
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], kde=True, bins=15, color='skyblue')
plt.title("Distribution of Final Grades (G3)")
plt.savefig("outputs/g3_distribution.png")
plt.close()

# Normal Curve Overlay
mean = df['G3'].mean()
std = df['G3'].std()
x = np.linspace(df['G3'].min(), df['G3'].max(), 100)
normal_curve = stats.norm.pdf(x, mean, std)

plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=15, stat='density', color='lightcoral', label='Actual')
plt.plot(x, normal_curve, 'k--', label='Normal Curve')
plt.legend()
plt.title("G3 vs Normal Distribution")
plt.savefig("outputs/g3_vs_normal.png")
plt.close()

# Normality Test
stat, p = stats.normaltest(df['G3'])
print(f"\n🧪 Normality Test (G3): stat={stat:.2f}, p={p:.4f} → {'Not normal' if p < 0.05 else 'Normal'}")

# Pass probability
prob_pass = df['pass'].mean()
print(f"\n🎯 Probability of Passing: {prob_pass:.3f} ({df['pass'].sum()}/{len(df)})")

# Bar chart
sns.countplot(x='pass', data=df, palette='pastel')
plt.xticks([0, 1], ['Fail (<10)', 'Pass (≥10)'])
plt.title("Pass vs Fail Count")
plt.savefig("outputs/pass_fail_bar.png")
plt.close()

# ---------- PHASE 3: CONFIDENCE INTERVALS ----------
# Mean CI
SE = std / np.sqrt(len(df))
margin = 1.96 * SE
print(f"\n📏 95% CI for Mean G3: [{mean - margin:.2f}, {mean + margin:.2f}]")

# Proportion CI
p_hat = prob_pass
SE_p = np.sqrt(p_hat * (1 - p_hat) / len(df))
margin_p = 1.96 * SE_p
print(f"✅ 95% CI for Proportion Passing: [{p_hat - margin_p:.3f}, {p_hat + margin_p:.3f}]")

# ---------- PHASE 4: HYPOTHESIS TESTING ----------
# Male vs Female
t_stat, p_val = stats.ttest_ind(df[df['sex'] == 'M']['G3'], df[df['sex'] == 'F']['G3'])
print(f"\n👨‍🎓 Male vs Female T-Test: t={t_stat:.2f}, p={p_val:.4f} → {'Significant' if p_val < 0.05 else 'Not significant'}")

# Internet effect
t_stat, p_val = stats.ttest_ind(df[df['internet'] == 'yes']['G3'], df[df['internet'] == 'no']['G3'])
print(f"🌐 Internet Access T-Test: t={t_stat:.2f}, p={p_val:.4f} → {'Significant' if p_val < 0.05 else 'Not significant'}")

# Chi-square: pass vs internet
chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df['pass'], df['internet']))
print(f"📊 Chi² Test (Pass vs Internet): χ²={chi2:.2f}, p={p:.4f} → {'Dependent' if p < 0.05 else 'Independent'}")

# ---------- PHASE 5: CORRELATION ----------
corr_matrix = df[features].corr(method='pearson')
print("\n📈 Pearson Correlation Matrix:")
print(corr_matrix.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# Scatter plots
sns.scatterplot(x='G1', y='G3', data=df)
plt.title("G1 vs G3")
plt.savefig("outputs/scatter_g1_g3.png")
plt.close()

sns.scatterplot(x='studytime', y='G3', data=df)
plt.title("Study Time vs G3")
plt.savefig("outputs/scatter_studytime_g3.png")
plt.close()

sns.scatterplot(x='absences', y='G3', data=df)
plt.title("Absences vs G3")
plt.savefig("outputs/scatter_absences_g3.png")
plt.close()
