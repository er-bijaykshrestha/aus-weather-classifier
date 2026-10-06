# 🌦️ Australian Rainfall Prediction Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/Notebook-FinalProject__AUSWeather.ipynb-brightgreen.svg)](FinalProject_AUSWeather.ipynb)

An end-to-end Machine Learning classification pipeline built with **Python** and **Scikit-Learn** to predict daily precipitation (`RainToday`: Yes/No) in the metropolitan Melbourne region using historical Bureau of Meteorology sensor observations.

---

## 📌 Project Overview & Objectives

* **Goal**: Accurately predict same-day rainfall based on past multi-variable atmospheric indicators while proactively eliminating data leakage.
* **Geographic Scope**: Localized observation cluster across **Melbourne**, **Melbourne Airport**, and **Watsonia** (~7,500 daily records).
* **Class Imbalance**: ~76.3% dry days (`No`) vs. ~23.7% rainy days (`Yes`).
* **Baseline Accuracy**: 76.3% (naive majority-class baseline).
* **Primary Classifier**: **Random Forest** optimized via Stratified K-Fold Cross-Validation, beating the baseline with an **84.0% test accuracy** and **0.81 macro precision**.



## 🏗️ Machine Learning Pipeline Architecture

Raw Weather Dataset (145k+ entries)
│
▼
Data Cleaning & Geospatial Filtering (Melbourne Cluster: 7,557 rows)
│
▼
Feature Engineering (Date ➔ Season, Leakage Mitigation Target Realignment)
│
▼
Stratified Train/Test Split (80/20, stratify=y)
│
▼
ColumnTransformer Pipeline
├── Numeric Transformer: StandardScaler (16 continuous metrics)
└── Categoric Transformer: OneHotEncoder(handle_unknown='ignore') (6 categorical features)
│
▼
GridSearchCV Hyperparameter Optimization (5-Fold StratifiedKFold)
├── Model 1: Random Forest Classifier
└── Model 2: Logistic Regression (Baseline Comparison)
│
▼
Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, Feature Importance)



## 💻 Key Source Code

### 1. Leakage Mitigation & Feature Engineering
```python
import pandas as pd

# Re-align rain indicators to avoid predicting today using same-day end-of-day metrics
df = df.rename(columns={
    'RainToday': 'RainYesterday',
    'RainTomorrow': 'RainToday'
})

# Engineer cyclical meteorological feature from observation timestamps
def date_to_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    return 'Spring'

df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns='Date')

# Define target variable and predictor feature set
X = df.drop(columns='RainToday')
y = df['RainToday']


# 2. Preprocessing & Random Forest Pipeline with Grid Search

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# 80/20 stratified split preserving natural 76/24 distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Automated feature type detection
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# ColumnTransformer assembling feature processors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Full scikit-learn training pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid for cross-validation
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)


# 3. Alternative Classifier: Logistic Regression Pipeline

from sklearn.linear_model import LogisticRegression

# Swap out classifier inside existing pipeline structure
rf_pipeline.set_params(classifier=LogisticRegression(random_state=42))

lr_param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

grid_search_lr = GridSearchCV(rf_pipeline, lr_param_grid, cv=cv, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)


# 📊 Evaluation & Results
Model Performance Comparison

Model	               Cross-Validation Score	   Test Accuracy	Precision (Rain: Yes)	Recall (Rain: Yes)	F1-Score (Rain: Yes)
Zero-Rule Baseline	          —	                        76.3%	         0.00	                  0.00	                 0.00
Logistic Regression	          83.0%	                    83.0%	         0.68	                  0.51	                 0.58
Random Forest (Optimized)	  85.1%	                    84.0%	         0.75	                  0.51	                 0.61


# Final Classification Report (Random Forest Estimator)

precision    recall  f1-score   support

          No       0.86      0.95      0.90      1154
         Yes       0.75      0.51      0.61       358

    accuracy                           0.84      1512
   macro avg       0.81      0.73      0.76      1512
weighted avg       0.84      0.84      0.83      1512


# 📈 Visual Outputs
1. Confusion Matrix

               Predicted: No     Predicted: Yes
Actual: No        1098               56
Actual: Yes        176              182


High true negative rate (95% specificity), effectively minimizing false alarms.

Precision of 75% for rainy days indicates that when the model forecasts rain, it is reliable 3 out of 4 times.

2. Atmospheric Feature Importance Rankings
Extracted directly from the best-performing Random Forest estimator:

1. Humidity3pm: Strongest individual indicator of same-day precipitation.
2. Sunshine: Hours of sunlight inversely correlated with storm activity.
3. Pressure3pm / Pressure9am: Barometric pressure shifts capturing incoming frontal systems.
4. Cloud3pm / Cloud9am: Sky cloud coverage ratio confirming low-pressure troughs.
5. WindGustSpeed Peak gust velocities driving storm fronts into the Melbourne basin.

📂 Repository Layout

├── FinalProject_AUSWeather.ipynb   # Complete Jupyter Notebook containing EDA, preprocessing, & models
├── LICENSE                         # MIT License
└── README.md                       # Project technical documentation

