# Heart Disease Prediction using Machine Learning

A machine learning project that predicts the presence or absence of heart disease using the **UCI Cleveland Heart Disease Dataset**. It implements four classification algorithms, optimizes their hyperparameters using Optuna, and provides an interactive Streamlit application to compare their predictions.

## Overview

The project explores how different machine learning algorithms perform on a relatively small medical dataset. Instead of relying on a single classifier, it trains and evaluates four models independently.

**Models implemented:**
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

Users can enter patient attributes through the Streamlit interface and compare predictions from all four models.

## Dataset

**Source:** [UCI Machine Learning Repository – Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

The project uses the processed Cleveland dataset.

- Original observations: 303
- Input features: 13
- Target: Presence or absence of heart disease

The original target contains values from 0 to 4. It is converted into a binary classification target:

- `0`: No heart disease
- `1`: Heart disease present

## Features

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia-related category |

## Methodology

### 1. Exploratory Data Analysis

- Examined feature distributions and relationships with the target.
- Investigated missing values and class distribution.
- Analysed numerical and categorical features.
- Identified potential patterns and relationships between patient attributes and heart disease.

### 2. Data Preprocessing

- Missing-value imputation.
- Feature transformation using Yeo-Johnson Power Transformation for selected numerical features.
- One-hot encoding of categorical features.
- Separate preprocessing pipelines for numerical and categorical variables.

### 3. Model Training

Four classification models were trained using Scikit-learn.

Different sampling strategies were explored:

| Model | Sampling Strategy |
|---|---|
| Logistic Regression | SMOTE |
| SVM | Baseline |
| Decision Tree | SMOTE |
| Random Forest | Baseline |

### 4. Hyperparameter Optimization

Optuna was used to tune the selected models.

- Optimization algorithm: Tree-structured Parzen Estimator (TPE)
- Trials per model: 50
- Cross-validation: 5-fold Stratified Cross-Validation
- Optimization objective: F1-score and recall, with greater emphasis on recall.

The best configurations were used to construct the final models.

## Model Evaluation

Models were evaluated on a held-out test set using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Test Set Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 86.89% | 81.25% | 92.86% | 86.67% | 96.00% |
| SVM | 81.97% | 75.76% | 89.29% | 81