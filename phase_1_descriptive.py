#Phase1: Descriptive 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os 

#Creates outputs folder if not 
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv('student-mat.csv', sep=';')
print(df.head())
#Descriptive Analysis
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

#Key Features Summary
features = ['age', 'absences', 'studytime', 'failures', 'G1', 'G2', 'G3']
print("\nAggregated Stats (mean, median, std, min, max): ")
print(df[features].agg(['mean', 'median', 'std', 'min', 'max']))

#Plots
sns.set_theme(style="whitegrid")

# Histogram: G3
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], kde=True, bins=15)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/histogram_g3.png")
plt.show()

# Boxplot: G3 by Gender
plt.figure(figsize=(6, 4))
sns.boxplot(x='sex', y='G3', data=df)
plt.title("Final Grade by Gender")
plt.tight_layout()
plt.savefig("outputs/boxplot_g3_gender.png")
plt.show()

print("\nPhase 1 completed. Charts saved in 'outputs/' folder.")




