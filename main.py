# Importing all required libraries
import pandas as pd            # For reading and analyzing data
import numpy as np             # For handling numerical operations
import plotly.express as px    # For creating interactive charts
from sklearn.model_selection import train_test_split  # For splitting data into training and testing parts
from sklearn.tree import DecisionTreeClassifier        # For building a simple machine learning model

# ----------------------------
# STEP 1: READING THE DATA
# ----------------------------
# The 'r' before the path tells Python to treat backslashes (\) as normal characters
try:
    data = pd.read_csv(r"D:\Documents\Projects\Python\Online Payment Fraud Detection\credit card.csv")
except FileNotFoundError:
    # If the file isn’t found, this block will run instead of crashing the program
    print("Error: 'credit card.csv' not found.")
    print("Please make sure the file is in the correct directory or provide the full absolute path.")
    exit()  # Stop running if data isn’t found

# Show first 5 rows of the dataset
print(data.head())

# Check if there are any missing values in the data
print(data.isnull().sum())

# ----------------------------
# STEP 2: VISUALIZE TRANSACTION TYPES
# ----------------------------
# Count how many times each transaction type appears
print(data.type.value_counts())
type_counts = data["type"].value_counts()  # Example: CASH_OUT = 10000, TRANSFER = 5000, etc.
transactions = type_counts.index           # The names of transaction types
quantity = type_counts.values              # The counts (numbers) for each type

# Create a donut-style pie chart showing transaction types
figure = px.pie(values=quantity, 
                 names=transactions, 
                 hole=0.5, 
                 title="Distribution of Transaction Type")
figure.show()

# ----------------------------
# STEP 3: VISUALIZE FRAUD CASES
# ----------------------------
# Count how many transactions are fraud (1) and non-fraud (0)
fraud_counts = data["isFraud"].value_counts()
# Convert 0 → "No Fraud" and 1 → "Fraud" for better readability
fraud_labels = fraud_counts.index.map({0: "No Fraud", 1: "Fraud"})
fraud_quantity = fraud_counts.values

# Create a pie chart showing how many fraud vs normal transactions exist
fraud_figure = px.pie(values=fraud_quantity, 
                      names=fraud_labels, 
                      hole=0.5, 
                      title="Fraud vs. No Fraud Distribution")

# Add percentages (like 98% No Fraud, 2% Fraud) inside the chart
fraud_figure.update_traces(textinfo='percent+label')
fraud_figure.show()

# ----------------------------
# STEP 4: CORRELATION CHECK
# ----------------------------
# Correlation tells how strongly one column is related to another
# numeric_only=True avoids text columns like 'type'
correlation = data.corr(numeric_only=True)
print(correlation["isFraud"].sort_values(ascending=False))  # Sort by most related to fraud

# ----------------------------
# STEP 5: CONVERT TEXT TO NUMBERS
# ----------------------------
# Machine learning models cannot understand text (like 'CASH_OUT'),
# so we replace them with numbers manually.
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})

# ----------------------------
# STEP 6: PREPARE DATA FOR MODEL
# ----------------------------
# x = inputs (features), y = output (what we want to predict)
# We choose only important columns for simplicity
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# Create a new readable column for humans to understand later
data["isFraud_mapped"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

# ----------------------------
# STEP 7: SPLIT DATA INTO TRAINING AND TESTING
# ----------------------------
# 90% data for training, 10% data for testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# ----------------------------
# STEP 8: TRAIN MACHINE LEARNING MODEL
# ----------------------------
# Decision Tree is a simple but effective algorithm
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)  # Model learns patterns from the training data
print(f"Model Score: {model.score(xtest, ytest)}")  # How accurate the model is

# ----------------------------
# STEP 9: TEST THE MODEL WITH EXAMPLES
# ----------------------------

# Function to convert numeric prediction (0/1) into readable text
def get_prediction_text(prediction):
    return "Fraud" if prediction[0] == 1 else "No Fraud"

# --- Test 1: A suspicious-looking transaction ---
fraud_features = np.array([[4, 9000.60, 9000.60, 0.0]])
fraud_prediction = model.predict(fraud_features)
print(f"\n--- Test Case 1 (Fraudulent) ---")
print(f"Features: {fraud_features}")
print(f"Prediction: {get_prediction_text(fraud_prediction)}")

# --- Test 2: A normal, genuine transaction ---
normal_features = np.array([[2, 9839.64, 170136.0, 160296.36]])
normal_prediction = model.predict(normal_features)
print(f"\n--- Test Case 2 (Normal) ---")
print(f"Features: {normal_features}")
print(f"Prediction: {get_prediction_text(normal_prediction)}")
