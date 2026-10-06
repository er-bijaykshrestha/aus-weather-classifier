# 🌦️ Australian Rainfall Prediction Classifier

An end-to-end Machine Learning classification pipeline built with **Python** and **Scikit-Learn** to predict whether it will rain on a given day in the Melbourne metropolitan area using historical Bureau of Meteorology weather observations.

---

## 📌 Executive Summary

* **Objective**: Predict daily precipitation (`RainToday`: Yes/No) using historical meteorological parameters while mitigating potential data leakage.
* **Target Geographic Area**: Melbourne, Melbourne Airport, and Watsonia regions.
* **Key Challenge**: Handling real-world imbalanced weather data (~76% No Rain / ~24% Rain) and high-dimensional categorical features.
* **Primary Models**: Random Forest Classifier & Logistic Regression with Cross-Validation Grid Search.

---

## 🛠️ Tech Stack & Libraries

* **Core Language**: Python 3
* **Data Manipulation & Analysis**: Pandas, NumPy
* **Machine Learning & Preprocessing**: Scikit-Learn
* **Data Visualization**: Matplotlib, Seaborn

---

## ⚙️ Machine Learning Pipeline Architecture

1. **Data Cleaning & Filtering**:
   - Filtered dataset to high-density Melbourne regional clusters (7,500+ records) for localized consistency.
   - Handled missing values across sensor metrics (pressure, humidity, temperature, wind).
2. **Feature Engineering**:
   - Extracted cyclical meteorological pattern variables (`Season`: Summer, Autumn, Winter, Spring) from timestamp observations.
   - Restructured temporal target variables to prevent future data leakage (`RainToday` as target using past observation features).
3. **Automated Preprocessing (`ColumnTransformer`)**:
   - **Continuous Features**: Scaled via `StandardScaler`.
   - **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown='ignore')`.
4. **Stratified Splitting & Validation**:
   - Implemented an 80/20 train/test split utilizing `stratify=y` to preserve the natural imbalanced class distribution.
   - Evaluated models using 5-fold Stratified Cross-Validation (`StratifiedKFold`).
5. **Hyperparameter Tuning (`GridSearchCV`)**:
   - Optimized tree depth, estimator counts, and leaf split constraints for `RandomForestClassifier`.
   - Tuned penalty configurations (`l1`, `l2`), class weighting (`balanced`), and solvers for `LogisticRegression`.

---

## 📊 Key Results & Evaluation

| Metric | Random Forest (Best Tuned) | Logistic Regression |
| :--- | :---: | :---: |
| **CV Accuracy** | ~85.0% | ~83.0% |
| **Test Accuracy** | ~84.0% | ~83.0% |
| **Precision (Rain: Yes)** | ~0.75 | ~0.68 |
| **Recall (Rain: Yes)** | ~0.51 | ~0.51 |

* Evaluated using Confusion Matrix analysis and classification metrics (precision, recall, F1-score) to ensure performance outperformed naive majority-class guessing (~76%).
* Derived tree-based feature importances across transformed atmospheric features (humidity at 3 PM and pressure metrics proved most decisive).

---

## 📂 Repository Structure

```text
├── FinalProject_AUSWeather.ipynb   # Complete analysis, training pipeline, and evaluation
├── LICENSE                         # MIT License
└── README.md                       # Project overview and technical documentation
