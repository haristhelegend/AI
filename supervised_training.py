# In this lab, we provide the "Label" (the answer key)
from sklearn.linear_model import LinearRegression

# DATA (The "Experience")
# Input: Study Hours | Target: Test Score (The Answer Key)
X = [[1], [2], [3], [4], [5]] 
y = [50, 60, 70, 80, 90]      # We are "supervising" by providing these answers

# MODEL
model = LinearRegression()

# TRAINING (The "Learning" phase)
model.fit(X, y)

# PREDICTION (Testing)
# We test it with a value it hasn't seen: 7 hours
new_input = [[7]]
prediction = model.predict(new_input)

print(f"Based on the training, a student studying 7 hours is predicted to score: {prediction[0]}")