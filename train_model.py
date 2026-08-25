import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Our Data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Test_Score': [50, 60, 70, 80, 90]
}
df = pd.DataFrame(data)

# 2. Reshape data for Scikit-Learn
# Scikit-learn expects a 2D array for the input (X)
X = df[['Hours_Studied']] 
y = df['Test_Score']

# 3. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 4. Predict
# Let's predict the score for 6 hours of study
hours = np.array([[6]])
prediction = model.predict(hours)

print(f"If you study for 6 hours, the predicted score is: {prediction[0]}")