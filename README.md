# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Surjith.D
RegisterNumber: 212223043006 
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Load dataset
data = pd.read_csv("/Salary.csv")

# Display initial information
print(data.head())
print(data.info())
print(data.isnull().sum())

# Encode categorical data
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

# Define features and target
x = data[["Position", "Level"]]
y = data["Salary"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Predict using the model
y_pred = dt.predict(x_test)
print("Predicted salaries:", y_pred)

# Evaluate the model
r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Predict for a custom input
custom_prediction = dt.predict([[5, 6]])
print("Custom prediction for Position 5, Level 6:", custom_prediction)
```

## Output:

![ex 9](https://github.com/user-attachments/assets/d1e617fa-621d-42c3-b33b-23e833caaba8)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
