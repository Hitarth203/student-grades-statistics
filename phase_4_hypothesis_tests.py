# phase_4_hypothesis_tests.py

import pandas as pd
import numpy as np
import scipy.stats as stats
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("student-mat.csv", sep=';')

# Create pass/fail column
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

#T-Test for Male vs Female grades 

male_grades = df[df['sex'] == 'M']['G3']
female_grades = df[df['sex'] == 'F']['G3']

t_stat, p_val = stats.ttest_ind(male_grades, female_grades)

print("\nHypothesis: Do male and female students score differently?")
print(f"T-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
if p_val < 0.05:
    print("Reject H₀: Significant difference in average grades by gender")
else:
    print("Fail to reject H₀: No significant difference in grades by gender")

#T-Test: Internet Access vs No Internet 

internet_yes = df[df['internet'] == 'yes']['G3']
internet_no = df[df['internet'] == 'no']['G3']

t_stat, p_val = stats.ttest_ind(internet_yes, internet_no)

print("\nHypothesis: Does internet access affect grades?")
print(f"T-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
if p_val < 0.05:
    print("Reject H₀: Internet access significantly affects grades")
else:
    print("Fail to reject H₀: No significant effect of internet access on grades")

#Chi-Square Test: Pass/fail vs Internet Access

contingency = pd.crosstab(df['pass'], df['internet'])

chi2, p, dof, expected = stats.chi2_contingency(contingency)

print("\nHypothesis: Is it related to internet access?")
print(f"Chi² = {chi2:.3f}, p-value = {p:.4f}")
if p < 0.05:
    print("Reject H₀: Internet access and passing are dependent")
else:
    print("Fail to reject H₀: No dependency between internet access and passing")
