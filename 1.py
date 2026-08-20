import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("churnguard_data.csv")
df.shape
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()
df["Churn"].value_counts(dropna=False)
df["Contract"].unique()
df.drop(columns=["customerID"], inplace=True, errors="ignore")
df.drop_duplicates(inplace=True)
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()
contract_map = {
    "Month-to-month": "Month-to-month",
    "Month To Month": "Month-to-month",
    "Month to Month": "Month-to-month",
    "Monthly": "Month-to-month",
    "Month": "Month-to-month",

    "One year": "One year",
    "1 year": "One year",
    "1 Year": "One year",
    "One Year": "One year",

    "Two year": "Two year",
    "2 year": "Two year",
    "2 Year": "Two year",
    "Two Year": "Two year"
}

df["Contract"] = df["Contract"].str.strip().replace(contract_map)
internet_map = {
    "dsl": "DSL",
    "DSL": "DSL",
    "fiberoptic": "Fiber optic",
    "FiberOptic": "Fiber optic",
    "Fiber optic": "Fiber optic",
    "Fibre optic": "Fiber optic",
    "fiber optic": "Fiber optic",
    "None": "No",
    "No": "No",
    "no": "No"
}
df["InternetService"] = df["InternetService"].str.strip().replace(internet_map)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df[df["tenure"] > 0]
df = df[(df["MonthlyCharges"] >= 10) & (df["MonthlyCharges"] <= 200)]
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
df["tenure"] = df["tenure"].fillna(round(df["tenure"].median())).astype(int)
print(df.shape)
print(df.isnull().sum())
df.drop(columns=["customerID"], inplace=True, errors="ignore")
df.drop_duplicates(inplace=True)
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()
contract_map = {
    "Month-to-month": "Month-to-month",
    "Month To Month": "Month-to-month",
    "Month to Month": "Month-to-month",
    "Monthly": "Month-to-month",
    "Month": "Month-to-month",
    "One year": "One year",
    "One Year": "One year",
    "1 year": "One year",
    "1 Year": "One year",
    "Two year": "Two year",
    "Two Year": "Two year",
    "2 year": "Two year",
    "2 Year": "Two year"
}
df["Contract"] = df["Contract"].str.strip().replace(contract_map)
internet_map = {
    "dsl": "DSL",
    "DSL": "DSL",
    "FiberOptic": "Fiber optic",
    "fiberoptic": "Fiber optic",
    "fiber optic": "Fiber optic",
    "Fibre optic": "Fiber optic",
    "Fiber optic": "Fiber optic",
    "None": "No",
    "No": "No",
    "no": "No"
}
df["InternetService"] = df["InternetService"].str.strip().replace(internet_map)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df[df["tenure"] > 0]
df = df[(df["MonthlyCharges"] >= 10) & (df["MonthlyCharges"] <= 200)]
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
df["tenure"] = df["tenure"].fillna(round(df["tenure"].median())).astype(int)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
categorical_cols = [
    "gender",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

df = pd.read_csv("churnguard_data.csv")
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}
df["Contract"] = df["Contract"].map(contract_map)
df.dropna(inplace=True)
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract"
]

X = df[features]
y = df["Churn"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
tenure = int(input("Enter tenure (months): "))
monthly = float(input("Enter Monthly Charges: "))
total = float(input("Enter Total Charges: "))
senior = int(input("Senior Citizen? (1 = Yes, 0 = No): "))
contract = int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "))
user_data = [[tenure, monthly, total, senior, contract]]
prediction = model.predict(user_data)[0]
if prediction == 1:
    print("Prediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")