# phase_5_correlation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("student-mat.csv", sep=';')

#Selecting Numeric Columns for Correlation 
features = ['age', 'absences', 'studytime', 'failures', 'G1', 'G2', 'G3']
df_corr = df[features]

#Pearson Correlation Matrix 
corr_matrix = df_corr.corr(method='pearson')
print("\nPearson Correlation Matrix:\n")
print(corr_matrix.round(2))

# Heatmap 
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# Scatter Plots 
# G1 vs G3
plt.figure(figsize=(6, 4))
sns.scatterplot(x='G1', y='G3', data=df, color='green')
plt.title("G1 vs Final Grade (G3)")
plt.tight_layout()
plt.savefig("outputs/scatter_g1_g3.png")
plt.show()

# Study time vs G3
plt.figure(figsize=(6, 4))
sns.scatterplot(x='studytime', y='G3', data=df, color='orange')
plt.title("Study Time vs Final Grade (G3)")
plt.tight_layout()
plt.savefig("outputs/scatter_studytime_g3.png")
plt.show()

# Absences vs G3
plt.figure(figsize=(6, 4))
sns.scatterplot(x='absences', y='G3', data=df, color='red')
plt.title("Absences vs Final Grade (G3)")
plt.tight_layout()
plt.savefig("outputs/scatter_absences_g3.png")
plt.show()
