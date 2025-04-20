# 🚧 Road Accident Survival Prediction using Logistic Regression

This project is built as part of an AI training program under FIIT Delhi, focusing on predicting the **survival chances of road accident victims** using **Logistic Regression**. The model is trained on a real-world dataset and uses accident-related factors to classify survival outcomes.

---

## 🎯 Goal

To develop a machine learning model using **Logistic Regression** to predict whether a victim will survive a road accident, based on features such as:

- Victim’s age
- Speed of the vehicle
- Seatbelt usage
- Emergency response time
- Weather and road conditions

---

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [EDA](#exploratory-data-analysis-eda)
4. [Model Training](#model-training)
5. [Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)
7. [Future Scope](#future-scope)
8. [Colab Notebook](#project-link)

---

## 🧠 Introduction

Road accidents are a major cause of death globally. By predicting the survival likelihood using logistic regression, this project helps emergency services, hospitals, and government bodies make timely and informed decisions.

### 📌 Why Logistic Regression?

- Simple yet effective for binary classification
- Interpretable and fast
- Performs well on clean, linearly separable data

---

## 📊 Dataset

- Real-world road accident data
- Features include: age, location, speed, weather, emergency response time, seatbelt usage
- Data preprocessing included:
  - Handling missing values
  - Encoding categorical variables
  - Scaling numerical features

---

## 🔍 Exploratory Data Analysis (EDA)

- Descriptive statistics
- Distribution plots & correlation heatmaps
- Box plots of features vs. survival
- Label Encoding of categorical data
- Null value handling (mean/mode)

---

## 🧪 Model Training

- Selected **Logistic Regression** for binary classification
- Steps:
  - Feature selection based on correlation
  - Train-test split (80-20)
  - Trained using `LogisticRegression()` from scikit-learn

---

## 📈 Model Evaluation

- Accuracy: **~85%**
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- K-Fold Cross Validation for robustness

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
