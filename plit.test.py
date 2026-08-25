from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Imagine this is a larger dataset
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [10, 20, 30, 40, 50, 60, 70, 80]

# We split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

# Train ONLY on the 80%
model.fit(X_train, y_train)

# Test on the 20% the model has never seen
score = model.score(X_test, y_test)

print(f"Model Accuracy Score: {score}")