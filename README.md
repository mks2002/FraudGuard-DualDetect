# 🛡️ FraudGuard: DualDetect for Transaction Fraud Detection

**FraudGuard-DualDetect** is an end-to-end binary classification system that detects fraudulent online transactions using engineered features, deep exploratory analysis, and optimized ML models. The project focuses on accuracy, explainability, and real-world fraud patterns.

---

## 📦 Dataset Overview

- **Rows**: 23,634 transactions  
- **Features**: 16  
- **Fraud Cases**: 1,222 (≈5.17%)  
- **Legit Cases**: 22,412 (≈94.83%)

### 🔑 Key Columns

- `Transaction Amount`, `Payment Method`, `Quantity`, `Customer Age`, `Device Used`, `Customer Location`
- `Transaction Hour`, `Account Age Days`, `IP Address`, `Shipping Address`, `Billing Address`
- `Is Fraudulent` (target label)

---

## 🔍 Workflow Summary

1. **EDA & Visualization**
   - 15+ plots including histograms, bar charts, boxplots, and time-based fraud patterns
   - Compared fraud vs legit transactions across devices, hours, and geolocation

2. **Feature Engineering**
   - Created 10+ new features: time gaps, location match flags, rare-device indicators
   - Encoded categorical features using target encoding and one-hot encoding
   - Used filter and wrapper methods to select top 8 most informative features

3. **Model Building**
   - Compared 5+ models:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - LightGBM
     - Gradient Boosting
   - Tuned hyperparameters using Grid Search & Randomized Search
   - Evaluated with stratified split and ROC-AUC for imbalanced learning

---

## ✅ Results

| Metric              | Value     |
|---------------------|-----------|
| **Test Accuracy**   | 94%       |
| **Precision (Fraud)** | 91%     |
| **Recall (Fraud)**  | 88%       |
| **F1-Score (Fraud)**| 89%       |
| **ROC AUC**         | 0.96+     |
| **Features Selected**| 8 of 16  |

> Class imbalance handled via stratified train-test splits and evaluation focused on recall & F1.

---

## 📊 Visual Insights

<p align="center">
  <img src="https://github.com/mks2002/FraudGuard-DualDetect/blob/main/Categorical-COLs.png" width="400" title="Categorical COLS"/>
  <img src="https://github.com/mks2002/FraudGuard-DualDetect/blob/main/Categorical-Cols-PIE.png" width="400" title="Categorical COLS PIE CHART"/>
  <img src="https://github.com/mks2002/FraudGuard-DualDetect/blob/main/Numerical-COLS-KDE.png" width="400" title="NUMERIC COLS"/>
  <img src="https://github.com/mks2002/FraudGuard-DualDetect/blob/main/Numerical-Target-BOX-PLOT.png" width="400" title="NUMERIC COLS VS TARGET Box Plot"/>
  <img src="https://github.com/mks2002/FraudGuard-DualDetect/blob/main/Precision-Recall-Threshold.png" width="400" title="Precision Recall Threshold"/>
</p>

---

## 🛠 Tech Stack

- Python (pandas, NumPy)
- Scikit-learn
- XGBoost, LightGBM
- Seaborn, Matplotlib

---

## 🗂 Directory Structure

```
📁 FraudGuard-DualDetect
├── Fraud-Transaction-1.ipynb
├──Fraud-Transaction-2.ipynb
├── README.md
└── Categorical-COLs.png
├── Categorical-Cols-PIE.png
├── Numerical-COLS-KDE.png
├── Numerical-Target-BOX-PLOT.png
└──  Precision-Recall-Threshold.png

```

---

## 🚀 How to Run

1. Clone the repo  
   `git clone https://github.com/yourusername/FraudGuard-DualDetect.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Launch notebook  
   `jupyter notebook fraud_detection_model.ipynb`

---

## 📌 Highlights

- Handled real-world fraud imbalance (5% fraud) without oversampling
- Built interpretable model pipeline with 94% overall accuracy
- Identified device + hour patterns as top fraud signals
- F1-score improved from **0.85 → 0.89** using tuned XGBClassifie

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

**Mayuk Sarkar** – [@mayuksarkar](https://github.com/mks2002)
