# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using classification models and K-Fold cross-validation. The dataset is highly imbalanced, making evaluation and feature importance analysis key components of this pipeline.

---

## 🎯 Goal

Build a reliable model to identify fraudulent transactions from anonymized credit card data.

---

## 📦 Dataset Summary

- Features: `Time`, `Amount`, `V1`–`V28` (PCA components)
- Target: `Class` (0 = Not Fraud, 1 = Fraud)
- Contains strong class imbalance (~0.17% fraud)

---

## 🔍 Key Steps

- Data cleaning and feature engineering (e.g., `Hour` from `Time`)
- Visual analysis: distribution plots, heatmaps, and boxplots
- Model training with:
  - Random Forest, AdaBoost, CatBoost, LightGBM, XGBoost
- Evaluation using ROC AUC, confusion matrices
- K-Fold cross-validation for robust model validation

---

## 🧰 Frameworks/Libraries: I used on this project

### 🧮 Data Manipulation & Analysis
- **Pandas** – for handling datasets and exploratory analysis  
- **NumPy** – efficient numerical operations  

### 📊 Data Visualization
- **Matplotlib** – basic visualizations  
- **Seaborn** – advanced statistical plots  
- **Plotly** – interactive browser-based plots  

### ⚙️ Machine Learning
- **Scikit-learn (sklearn)** – ML models and utilities:  
  `train_test_split`, `KFold`, `roc_auc_score`, `RandomForestClassifier`, `AdaBoostClassifier`, `svm`  
- **CatBoost** – efficient gradient boosting with categorical support  
- **LightGBM** – fast, scalable gradient boosting  
- **XGBoost** – highly optimized gradient boosting  

### 🕒 Other Utilities
- **gc** – manual memory management during training  
- **datetime** – handling time-related features  

---

## 📊 Evaluation Metrics

- ROC AUC Score  
- Confusion Matrix  
- Feature Importance

---

## 📁 Structure

