# Healthcare ML - Diabetes Prediction

import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

data = pd.read_csv(url)

# Display the first five rows
print(data.head())

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

print("Features:")
print(X.head())

print("Target:")
print(y.head())
