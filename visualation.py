import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([50, 60, 70, 80, 90])

# Train
model = LinearRegression()
model.fit(X, y)

# Predict for the plot
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue') # Real data
plt.plot(X, y_pred, color='red') # The "Line of Best Fit"
plt.xlabel('Hours')
plt.ylabel('Score')
# Save the plot instead of trying to open a window
plt.savefig('my_first_graph.png')
print("Graph saved successfully as 'my_first_graph.png'!")