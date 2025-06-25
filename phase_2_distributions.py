#Phase 2: Distributions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv("student-mat.csv", sep=';')

#Normal Distribution Curve 
mean = df['G3'].mean()
std = df['G3'].std()
x = np.linspace(df['G3'].min(), df['G3'].max(), 100)
normal_curve = stats.norm.pdf(x, mean, std)

plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=15, kde=False, stat='density', color='lightcoral', label='Actual')
plt.plot(x, normal_curve, 'k--', label='Normal PDF')
plt.title("G3 Distribution vs Normal Curve")
plt.xlabel("G3")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/g3_vs_normal.png")
plt.show()

#Normality Test 
stat, p = stats.normaltest(df['G3'])
print(f"\n🧪 Normality test for G3:\nStatistic = {stat:.3f}, p-value = {p:.4f}")
if p < 0.05:
    print("G3 is significantly not normally distributed (reject H₀)")
else:
    print("G3 seems normally distributed (fail to reject H₀)")

# 4. Create binary column: pass/fail
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# 5. Probability: How many passed?
total_students = len(df)
num_passed = df['pass'].sum()
prob_pass = num_passed / total_students

print(f"\nProbability of passing (G3 ≥ 10): {prob_pass:.3f} ({num_passed}/{total_students})")
