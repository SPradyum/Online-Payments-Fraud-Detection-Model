# 💳 Online Payment Fraud Detection using Machine Learning

> A Python-based project that detects fraudulent online payment transactions using a Decision Tree Classifier and interactive data visualizations.

---

## 🧠 Overview

Online financial fraud is a growing concern in digital payment systems.  
This project demonstrates how **machine learning** can identify potentially **fraudulent transactions** by analyzing key transactional features such as amount, balance changes, and transaction type.

The system not only builds a fraud detection model but also provides insightful **data visualizations** to understand transaction behavior.

---

## ✨ Features

- 📊 **Interactive Visualizations**
  - Transaction type distribution chart  
  - Fraud vs. Non-Fraud percentage chart  

- 🤖 **Machine Learning Model**
  - Uses a **Decision Tree Classifier**
  - Learns from real-world transaction data  

- 🧾 **Data Insights**
  - Correlation analysis to find which features relate most to fraud  
  - Fraud trend visualization  

- ✅ **User-Friendly Output**
  - Clear predictions (e.g., *Fraud* or *No Fraud*)  
  - Built-in error handling for missing files or invalid paths  

---

## 📂 Dataset Information

The project uses a CSV file named **`credit card.csv`**.

### Expected Columns:
| Column Name      | Description |
|------------------|-------------|
| `type`           | Transaction type (e.g., CASH_OUT, TRANSFER, PAYMENT) |
| `amount`         | Amount involved in the transaction |
| `oldbalanceOrg`  | Balance before the transaction |
| `newbalanceOrig` | Balance after the transaction |
| `isFraud`        | 1 for Fraudulent, 0 for Non-Fraudulent |

> ⚠️ **Note:** Make sure to update the dataset path inside the code:
```python
data = pd.read_csv(r"_path_\credit card.csv")
