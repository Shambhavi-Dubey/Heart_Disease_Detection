# Heart Disease Detection using Machine Learning

A full-fledged machine learning project to detect heart disease based on patient health metrics. This project uses classical machine learning models optimized with advanced techniques to deliver high accuracy, precision, and robustness. It includes a GUI application for easy predictions using trained models.

## 🩺 Project Overview

Heart disease is one of the leading causes of death globally. Early detection through predictive modeling can save lives. This project uses medical datasets and several machine learning algorithms to predict the presence of heart disease.


---

## 💾 Dataset Used

Merged and preprocessed data from:
- Cleveland
- Hungarian
- Long Beach VA
- Switzerland

Features include: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia, and target class.

---

## ⚙️ Features Implemented

- Missing value imputation
- Label encoding of categorical features
- Standardization of numerical features
- EDA (correlation heatmaps, boxplots)
- Dataset balancing using ADASYN
- Train/test split using stratification
- Model building with:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation (Accuracy, Precision, Recall, F1 Score, ROC AUC)
- Hyperparameter tuning using Optuna
- Exported models using Pickle
- GUI app for real-time prediction

---

## 🔍 Model Performance (Optimized)

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|----------|
| Logistic Regression| 0.7921   | 0.8125    | 0.7647 | 0.7879   | 0.8790   |
| Random Forest       | 0.8564   | 0.8687    | 0.8431 | 0.8557   | 0.9037   |
| XGBoost             | 0.8416   | 0.8571    | 0.8235 | 0.8400   | 0.8927   |

---

## 💡 Requirements

- Python 3.8+
- Libraries:
  - scikit-learn
  - xgboost
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - optuna
  - imbalanced-learn
  - tkinter (comes with Python)

Author
Shambhavi Dubey<br>
Final Year B.Tech CSE Student

Notes<br>
Models trained on a balanced dataset using ADASYN<br>
Optuna used for hyperparameter optimization<br>
GUI designed for simplicity and usability<br>
