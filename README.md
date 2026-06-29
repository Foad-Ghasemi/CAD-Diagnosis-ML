# 🫀 CAD Diagnosis — ML Pipeline

> **Machine Learning for Pre-Diagnosis Coronary Artery Disease Detection**  
> Feature Selection + Ensemble-based Decision Threshold Optimization via ROC Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project implements the machine learning pipeline developed as part of my M.Sc. thesis and two peer-reviewed publications on coronary artery disease (CAD) diagnosis using data mining techniques.

The pipeline combines:
1. **Heuristic Feature Selection** — combining Information Gain Ratio and Gini Index to identify the most discriminative clinical features
2. **Multi-classifier Benchmarking** — evaluating Decision Tree, Random Forest, AdaBoost, Gradient Boosting, SVM, k-NN, and Logistic Regression
3. **Ensemble-based Decision Threshold Optimization** — using ROC analysis (Youden's J statistic) to find the optimal classification threshold that balances sensitivity and specificity in clinical settings

---

## 📄 Publications

> **[1]** Ghasemi, F., et al. *"Feature Selection in Pre-Diagnosis Heart Coronary Artery Disease Detection: A Heuristic Approach Based on Information Gain Ratio and Gini Index."* 6th International Conference on Web Research (ICWR), IEEE, 2020.  
> DOI: [10.1109/ICWR49608.2020.9122285](https://doi.org/10.1109/ICWR49608.2020.9122285)

> **[2]** Ghasemi, F., et al. *"Improving Pre-Diagnosis Coronary Artery Disease Detection using Ensemble-based Decision Threshold in ROC Analysis."* Computer Methods and Programs in Biomedicine (Under Review).

---

## 🗂️ Project Structure

```
heart-disease-ml/
├── heart_disease_detection.py   # Main ML pipeline
├── cleveland_heart_disease.csv  # Dataset (Cleveland Heart Disease)
├── results.png                  # Output: plots (feature scores, ROC, comparison)
├── benchmark_results.csv        # Output: classifier performance table
├── requirements.txt             # Dependencies
└── README.md
```

---

## 📊 Dataset

**Cleveland Heart Disease Dataset** (UCI Machine Learning Repository)

| Property | Value |
|---|---|
| Samples | 303 |
| Features | 13 clinical features |
| Target | Binary (0 = no disease, 1 = disease) |
| Source | [UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |

**Features:** age, sex, chest pain type (cp), resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate (thalach), exercise-induced angina (exang), ST depression (oldpeak), slope, number of vessels (ca), thal

> **To use the real dataset:** Download `processed.cleveland.data` from the UCI repository and save it as `cleveland_heart_disease.csv` with the column names above.

---

## ⚙️ Methodology

### 1. Feature Selection

A heuristic scoring method combining two complementary criteria:

- **Information Gain Ratio (IGR)** — measures statistical dependency between feature and class label
- **Gini Index** — measures class impurity reduction

Combined score: `score = (IGR_normalized + Gini_normalized) / 2`

Top-8 features are selected for model training.

### 2. Classifier Benchmarking

All classifiers are evaluated using **5-fold stratified cross-validation** with metrics:
- Accuracy, AUC-ROC, Sensitivity, Specificity

### 3. Ensemble Threshold Optimization

In clinical settings, the default 0.5 threshold is suboptimal — a missed diagnosis (false negative) is far more costly than a false alarm.

The pipeline:
1. Averages probability predictions from all classifiers (ensemble)
2. Applies **Youden's J statistic** on the ROC curve to find the optimal threshold: `J = Sensitivity + Specificity - 1`
3. Applies the optimal threshold to the ensemble predictions

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python heart_disease_detection.py
```

### Output

- `results.png` — three plots: feature scores, classifier comparison, ROC curve with optimal threshold
- `benchmark_results.csv` — performance table for all classifiers

---

## 📈 Results

| Metric | Value |
|---|---|
| Ensemble AUC | ~0.73 |
| Optimal Threshold | ~0.55 |
| Sensitivity | ~0.83 |
| Specificity | ~0.70 |

> Results may vary slightly depending on the dataset version used (UCI original vs. synthetic demo).

---

## 👤 Author

**Foad Ghasemi**  
M.Sc. Computer Software Engineering — Shahid Ashrafi Esfahani University  
📧 foadghasemi1372@gmail.com  
🔗 [Google Scholar](https://scholar.google.com/citations?user=9XH7tgEAAAAJ)  
🔗 [LinkedIn](https://linkedin.com/in/foad-ghasemi-250b43173)

---

## 📝 License

This project is licensed under the MIT License.
