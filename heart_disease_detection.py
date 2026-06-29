"""
Heart Disease Detection using Feature Selection and Ensemble Methods
====================================================================
Author: Foad Ghasemi
Description:
    This project implements a machine learning pipeline for pre-diagnosis
    coronary artery disease (CAD) detection using:
    1. Heuristic Feature Selection (Information Gain Ratio + Gini Index)
    2. Ensemble-based Decision Threshold Optimization via ROC Analysis
    3. Multiple classifier benchmarking

Dataset: UCI Heart Disease / Cleveland Dataset
Reference Publications:
    [1] Ghasemi et al., "Feature Selection in Pre-Diagnosis Heart CAD Detection:
        A Heuristic Approach Based on Information Gain Ratio and Gini Index",
        IEEE ICWR, 2020. DOI: 10.1109/ICWR49608.2020.9122285
    [2] Ghasemi et al., "Improving Pre-Diagnosis CAD Detection using
        Ensemble-based Decision Threshold in ROC Analysis",
        Computer Methods and Programs in Biomedicine (Under Review).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────

def load_cleveland_dataset():
    """
    Load the Cleveland Heart Disease dataset.
    Download from: https://archive.ics.uci.edu/dataset/45/heart+disease
    Save as 'cleveland_heart_disease.csv' in the project directory.
    """
    import os
    csv_path = 'cleveland_heart_disease.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback: try UCI URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        df = pd.read_csv(url, names=columns, na_values='?')
        df.dropna(inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
        df.to_csv(csv_path, index=False)

    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    return df


# ──────────────────────────────────────────────
# 2. FEATURE SELECTION
# ──────────────────────────────────────────────

def compute_gini_index(X, y):
    """Compute Gini Index for each feature."""
    gini_scores = {}
    for col in X.columns:
        classes = np.unique(y)
        gini = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            gini -= p ** 2
        gini_scores[col] = gini
    return gini_scores


def heuristic_feature_selection(X, y, top_k=8):
    """
    Heuristic Feature Selection combining:
    - Information Gain Ratio (via mutual_info_classif as proxy)
    - Gini Index

    Selects top_k features based on combined normalized score.
    Based on: Ghasemi et al., IEEE ICWR 2020.
    """
    # Information Gain (using mutual information as proxy for IGR)
    ig_scores = mutual_info_classif(X, y, random_state=42)
    ig_normalized = ig_scores / (ig_scores.max() + 1e-10)

    # Gini Index
    gini_scores = compute_gini_index(X, y)
    gini_values = np.array([gini_scores[col] for col in X.columns])
    gini_normalized = gini_values / (gini_values.max() + 1e-10)

    # Combined heuristic score
    combined_scores = (ig_normalized + gini_normalized) / 2.0

    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'IG_score': ig_normalized,
        'Gini_score': gini_normalized,
        'combined_score': combined_scores
    }).sort_values('combined_score', ascending=False)

    selected_features = feature_scores.head(top_k)['feature'].tolist()

    print(f"\n{'='*50}")
    print("FEATURE SELECTION RESULTS")
    print(f"{'='*50}")
    print(feature_scores.to_string(index=False))
    print(f"\nSelected top-{top_k} features: {selected_features}")

    return selected_features, feature_scores


# ──────────────────────────────────────────────
# 3. CLASSIFIERS
# ──────────────────────────────────────────────

def get_classifiers():
    """Return dictionary of classifiers to benchmark."""
    return {
        'Decision Tree':      DecisionTreeClassifier(random_state=42),
        'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost':           AdaBoostClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM':                SVC(probability=True, random_state=42),
        'k-NN':               KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression':LogisticRegression(max_iter=1000, random_state=42),
    }


# ──────────────────────────────────────────────
# 4. ENSEMBLE THRESHOLD OPTIMIZATION (ROC-based)
# ──────────────────────────────────────────────

def find_optimal_threshold_roc(y_true, y_proba):
    """
    Find optimal decision threshold using ROC analysis.
    Maximizes Youden's J statistic (sensitivity + specificity - 1).
    Based on: Ghasemi et al., Computer Methods and Programs in Biomedicine (Under Review).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, fpr, tpr, thresholds


def ensemble_threshold_optimization(classifiers, X, y, cv=5):
    """
    Ensemble-based decision threshold optimization:
    1. Get probability predictions from all classifiers via cross-validation
    2. Average probabilities (ensemble)
    3. Find optimal threshold via ROC analysis
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    ensemble_proba = np.zeros(len(y))
    n_classifiers = 0

    print(f"\n{'='*50}")
    print("ENSEMBLE THRESHOLD OPTIMIZATION")
    print(f"{'='*50}")

    for name, clf in classifiers.items():
        proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
        ensemble_proba += proba
        n_classifiers += 1
        auc = roc_auc_score(y, proba)
        print(f"  {name:25s} | AUC: {auc:.4f}")

    ensemble_proba /= n_classifiers
    ensemble_auc = roc_auc_score(y, ensemble_proba)
    optimal_threshold, fpr, tpr, thresholds = find_optimal_threshold_roc(y, ensemble_proba)

    print(f"\n  {'Ensemble (avg)':25s} | AUC: {ensemble_auc:.4f}")
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.4f}")

    return ensemble_proba, optimal_threshold, fpr, tpr, ensemble_auc


# ──────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────

def evaluate_classifiers(classifiers, X, y, cv=5):
    """Benchmark all classifiers with 5-fold CV."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    print(f"\n{'='*50}")
    print("CLASSIFIER BENCHMARKING (5-Fold CV)")
    print(f"{'='*50}")
    print(f"{'Classifier':25s} | {'Accuracy':8s} | {'AUC':8s} | {'Sensitivity':11s} | {'Specificity':11s}")
    print("-" * 75)

    for name, clf in classifiers.items():
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        y_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        results.append({
            'Classifier': name,
            'Accuracy': acc,
            'AUC': auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })

        print(f"{name:25s} | {acc:.4f}   | {auc:.4f}   | {sensitivity:.4f}      | {specificity:.4f}")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 6. VISUALIZATION
# ──────────────────────────────────────────────

def plot_results(feature_scores, results_df, fpr, tpr, ensemble_auc, optimal_threshold, output_dir='.'):
    """Generate and save all result plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Feature Importance
    top_features = feature_scores.head(8)
    axes[0].barh(top_features['feature'], top_features['combined_score'], color='steelblue')
    axes[0].set_xlabel('Combined Score (IGR + Gini)', fontsize=11)
    axes[0].set_title('Feature Selection Scores\n(Information Gain Ratio + Gini Index)', fontsize=12)
    axes[0].invert_yaxis()

    # Plot 2: Classifier Comparison
    x = np.arange(len(results_df))
    width = 0.25
    axes[1].bar(x - width, results_df['Accuracy'], width, label='Accuracy', color='steelblue')
    axes[1].bar(x, results_df['AUC'], width, label='AUC', color='darkorange')
    axes[1].bar(x + width, results_df['Sensitivity'], width, label='Sensitivity', color='green')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['Classifier'], rotation=30, ha='right', fontsize=9)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title('Classifier Comparison (5-Fold CV)', fontsize=12)
    axes[1].legend()

    # Plot 3: Ensemble ROC Curve with optimal threshold
    axes[2].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'Ensemble ROC (AUC = {ensemble_auc:.4f})')
    axes[2].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    # Mark optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    axes[2].scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5,
                   label=f'Optimal threshold = {optimal_threshold:.3f}')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('Ensemble ROC Curve\n(Threshold Optimization via Youden\'s J)', fontsize=12)
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nResults plot saved to {output_dir}/results.png")


# ──────────────────────────────────────────────
# 7. MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CAD DIAGNOSIS — ML PIPELINE")
    print("Foad Ghasemi | github.com/Foad-Ghasemi")
    print("=" * 60)

    # Load data
    df = load_cleveland_dataset()
    X = df.drop('target', axis=1)
    y = df['target']

    # Feature Selection
    selected_features, feature_scores = heuristic_feature_selection(X, y, top_k=8)
    X_selected = X[selected_features]

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_selected), columns=selected_features)

    # Get classifiers
    classifiers = get_classifiers()

    # Benchmark all classifiers
    results_df = evaluate_classifiers(classifiers, X_scaled, y)

    # Ensemble threshold optimization
    ensemble_proba, optimal_threshold, fpr, tpr, ensemble_auc = \
        ensemble_threshold_optimization(classifiers, X_scaled, y)

    # Apply optimal threshold
    y_pred_optimal = (ensemble_proba >= optimal_threshold).astype(int)
    acc_optimal = accuracy_score(y, y_pred_optimal)
    cm = confusion_matrix(y, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print("FINAL RESULTS WITH OPTIMAL THRESHOLD")
    print(f"{'='*50}")
    print(f"Optimal Threshold : {optimal_threshold:.4f}")
    print(f"Ensemble AUC      : {ensemble_auc:.4f}")
    print(f"Accuracy          : {acc_optimal:.4f}")
    print(f"Sensitivity       : {tp/(tp+fn):.4f}")
    print(f"Specificity       : {tn/(tn+fp):.4f}")
    print(f"\n{classification_report(y, y_pred_optimal, target_names=['No Disease', 'Disease'])}")

    # Save plots
    plot_results(feature_scores, results_df, fpr, tpr, ensemble_auc,
                 optimal_threshold, output_dir='.')

    # Save results CSV
    results_df.to_csv('benchmark_results.csv', index=False)
    print("Benchmark results saved to benchmark_results.csv")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
