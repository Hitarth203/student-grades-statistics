#Phase 3: CI
import pandas as pd
import numpy as np
import scipy.stats as stats
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("student-mat.csv", sep=';')

# Create pass column again
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

#Confidence Interval for Mean G3

#Calculating mean and standard error
mean_g3 = df['G3'].mean()
std_g3 = df['G3'].std()
n = len(df)
SE = std_g3 / np.sqrt(n)  

# 95% Confidence Interval
z_score = 1.96  # This z_score is for 95% confidence
margin_error = z_score * SE

CI_lower = mean_g3 - margin_error
CI_upper = mean_g3 + margin_error

print(f"\n📏 95% Confidence Interval for Mean G3:")
print(f"Mean: {mean_g3:.2f}, SE: {SE:.2f}, Margin of Error: ±{margin_error:.2f}")
print(f"CI: [{CI_lower:.2f}, {CI_upper:.2f}]")

#Confidence Interval for Proportion of Passing Students

p_hat = df['pass'].mean() # sample proportion
n_pass = len(df)

SE_p = np.sqrt((p_hat * (1 - p_hat)) / n_pass)
margin_error_p = z_score * SE_p

CI_prop_lower = p_hat - margin_error_p
CI_prop_upper = p_hat + margin_error_p

print(f"\n95% Confidence Interval for Proportion of Students Passing:")
print(f"Proportion: {p_hat:.3f}, SE: {SE_p:.3f}, Margin of Error: ±{margin_error_p:.3f}")
print(f"CI: [{CI_prop_lower:.3f}, {CI_prop_upper:.3f}]")
