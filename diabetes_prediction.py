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
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
