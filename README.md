# 🏦 Loan Default Prediction
### Can we predict which customers will fail to repay their loans?

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Project Overview

Banks and financial institutions face significant risk when customers fail to repay their loans. This project builds a machine learning classification model to **predict loan defaults** based on customer financial and personal data — helping banks make smarter, data-driven lending decisions.

> **Business Goal:** Identify high-risk customers before approving loans to reduce financial losses.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle — Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) |
| Rows | 255,347 customers |
| Features | 16 (numeric + categorical) |
| Target | `Default` → 0 = Repaid ✅ / 1 = Defaulted ❌ |
| Class Balance | Imbalanced — 88.39% No Default / 11.61% Default |

---

## 🔍 Key Challenge — Imbalanced Classes

```
No Default (0): 225,694 customers  ████████████████████  88.39%
Default    (1):  29,653 customers  ██                    11.61%
```

A naive model predicting "No Default" for everyone achieves **88% accuracy** — but detects **0% of actual defaulters**. This project addresses this challenge using `class_weight='balanced'` and optimizing for **Recall** instead of Accuracy.

---

## 🛠️ Technical Stack

```python
Languages  : Python 3.10
Libraries  : Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
Environment: Google Colab
Workflow   : CRISP-DM
```

---

## 🔄 CRISP-DM Workflow

```
1. Business Understanding  →  Define the problem & success metric
2. Data Understanding      →  EDA, missing values, class balance
3. Data Preparation        →  Preprocessing pipeline (scaling + encoding)
4. Modeling                →  3 models × (default + tuned with GridSearchCV)
5. Evaluation              →  Classification report, confusion matrix, ROC/AUC
6. Conclusion              →  Model recommendation with justification
```

---

## 📈 Models Built

| Model | Default Recall (1) | Tuned Recall (1) | AUC |
|-------|-------------------|-----------------|-----|
| Logistic Regression | 0.03 | **0.69** ✅ | **0.75** |
| Random Forest | 0.01 | 0.50 | 0.74 |
| KNN | 0.07 | 0.06 | 0.61 |

### Tuning Strategy
- **GridSearchCV** with `cv=3` and `scoring='recall_macro'`
- Tested: solver, penalty (L1/L2/Elasticnet), C values, class_weight
- Models saved with `joblib` to avoid re-training

---

## 🏆 Best Model — Logistic Regression

```
Recall (Default)    : 0.69  →  Detects 69% of actual defaulters
Recall (No Default) : 0.68
Macro Avg Recall    : 0.68
AUC                 : 0.75
Best Params         : penalty=elasticnet, solver=saga, class_weight=balanced
```

### Why Logistic Regression?
- ✅ Highest Recall for Default class (0.69)
- ✅ Highest AUC (0.75)
- ✅ Interpretable — banks can explain loan rejections
- ✅ Fast training and prediction

---

## 📁 Project Structure

```
Loan-Default-Prediction/
│
├── Loan_default_project.ipynb   # Main notebook (full analysis)
├── README.md                    # Project documentation
└── data/
    └── Loan_default.csv         # Dataset (from Kaggle)
```

---

## 🚀 How to Run

```python
# 1. Clone the repository
git clone https://github.com/dohaalnabahin/Loan-Default-Prediction.git

# 2. Open in Google Colab or Jupyter Notebook

# 3. Install requirements
pip install pandas numpy scikit-learn seaborn matplotlib joblib

# 4. Run all cells in order
```

---

## 💡 Key Learnings

- Accuracy is **misleading** with imbalanced datasets → always check Recall
- `class_weight='balanced'` dramatically improves minority class detection
- Logistic Regression can outperform complex models on financial tabular data
- Saving models with `joblib` saves hours of re-training time

---

## 📬 Connect

**Doha Al-Nabahin**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/doha-samir12)

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/dohaalnabahin)

---

*Loan Default Prediction using Machine Learning | April 2026*
