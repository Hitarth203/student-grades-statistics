# Student Performance Analysis Project

This project performs a full statistical analysis on student performance using the client Student Performance Dataset. It walks through all essential statistical phases — from descriptive summaries to hypothesis testing and correlation analysis.

---

## Skills Used

- Descriptive Statistics (mean, median, std, skew, kurtosis)
- Probability Distributions
- Confidence Intervals
- Hypothesis Testing (t-test, chi-square)
- Correlation & Causation
- Data Visualization (Matplotlib & Seaborn)

---

## Dataset Info

- **Source**: Client work (can be shared)
- **File**: `student-mat.csv`
- **Size**: 649 students × ~30 features
- **Features**:
  - Demographics: sex, age, address
  - Academics: grades (G1, G2, G3), studytime, failures
  - Social: internet access, family support, absences

---

## Phases Covered

| Phase | Focus |
|-------|-----------------------------|
| 1     | Descriptive Statistics (mean, median, std, skew, kurtosis) |
| 2     | Distributions & Probability (normal curve, pass probability) |
| 3     | Confidence Intervals for mean and proportion |
| 4     | Hypothesis Testing (t-tests, chi-square) |
| 5     | Correlation & Causation (Pearson matrix, heatmap, scatter plots) |

---

## 🔍 Example Outputs

### Correlation Heatmap
Shows how strongly variables like G1, G2, studytime, and failures relate to the final grade (G3).

![Correlation Heatmap](outputs/correlation_heatmap.png)

---

### G3 Grade Distribution vs Normal Curve

![G3 Distribution](outputs/g3_vs_normal.png)

---

## Installation & Requirements

> Requires Python 3.7+

Install required packages:

```bash
pip install pandas numpy seaborn matplotlib scipy

