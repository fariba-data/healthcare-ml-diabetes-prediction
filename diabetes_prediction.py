# Healthcare ML - Diabetes Prediction

import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

data = pd.read_csv(url)

print(data.head())
