# Heart Disease Prediction Project

## Overview

This project delves into predicting heart disease using machine learning and data analysis techniques. The dataset, sourced from the UCI Machine Learning Repository and Kaggle, encompasses various medical attributes. The primary objective is to construct a robust model capable of accurately classifying individuals as either having or not having heart disease.

## Exploratory Data Analysis (EDA)

### Problem Definition

The central question is whether clinical parameters can predict the presence of heart disease in a patient.

### Data

The dataset comprises features such as age, sex, chest pain type, blood pressure, cholesterol levels, and more. Original sources include the UCI Machine Learning Repository and Kaggle.

### Evaluation

The aim is to achieve 95% accuracy during the proof of concept.

### Features

A detailed data dictionary outlines each feature, including age, sex, chest pain type, blood pressure, cholesterol levels, fasting blood sugar, and more.

### Data Exploration

EDA involves:
1. Analyzing the distribution of heart disease cases based on sex and chest pain type.
2. Investigating the relationship between age and maximum heart rate.
3. Assessing heart disease frequency per chest pain type.
4. Creating a correlation matrix to identify feature relationships.

## Importing Tools

Libraries such as pandas, matplotlib, seaborn, and scikit-learn are employed for data manipulation, visualization, and model implementation.

## Loading Data

The dataset is loaded, and its shape is confirmed.

```python
df = pd.read_csv("./heart-disease.csv")
df.shape  # (rows, columns)
```

## Data Exploration (EDA) Continued

### Heart Disease Frequency per Chest Pain Type

A detailed analysis of heart disease frequency based on chest pain type is conducted, involving cross-tabulation and visualization.

```python
pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(10, 6), color=["salmon", "lightblue"])
```

### Correlation Matrix

A correlation matrix is generated to visualize the relationships between different features.

## Modeling

### Data Preparation

The dataset is split into features (X) and the target variable (y). A train-test split is performed.

```python
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Model Comparison

Various machine learning models, including Logistic Regression, K-Nearest Neighbors, and Random Forest Classifier, are evaluated and compared.

```python
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)
```

### Hyperparameter Tuning

Hyperparameter tuning is performed using RandomizedSearchCV for Logistic Regression and RandomForestClassifier.

```python 
# Logistic Regression
rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, cv=5, n_iter=20, verbose=True)
rs_log_reg.fit(X_train, y_train)
rs_log_reg.best_params_

# Random Forest Classifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True)
rs_rf.fit(X_train, y_train)
rs_rf.best_params_
```

### Model Evaluation

Evaluation metrics, including ROC Curve, AUC Score, Confusion Matrix, and Classification Report, are generated.

```python 
RocCurveDisplay.from_estimator(gs_log_reg, X_test, y_test)
print(confusion_matrix(y_test, y_preds))
plot_conf_mat(y_test, y_preds)
print(classification_report(y_test, y_preds))
```
### Cross-Validated Metrics

Cross-validated metrics are calculated for accuracy, precision, recall, and F1-score.

```python
clf = LogisticRegression(C=0.20433597178569418, solver="liblinear")
cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision")
cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall")
cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1")
```
## Feature Importance

Feature importance is explored for the Logistic Regression model

```python
clf = LogisticRegression(C=0.20433597178569418, solver="liblinear")
cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
cv_precision = cross_val_score(clf, X, y, cv=5, scoring="precision")
cv_recall = cross_val_score(clf, X, y, cv=5, scoring="recall")
cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1")
```
## Results

* Achieved ~84% accuracy with the Logistic Regression model.
* Suggestions for improvement include collecting more data and exploring alternative models (e.g., CatBoost, XGBoost).
